# Bachelor's Thesis: Real-time Speech Translation System

Vedúci záverečnej práce: Ing. Ivan Minárik

## Abstract

This bachelor's thesis presents the design, implementation, and evaluation of a real-time live speech translation system. The project addresses the growing need for seamless cross-lingual communication, particularly in dynamic environments such as international conferences. Leveraging state-of-the-art open-source models for Speech-to-Text (STT), Machine Translation (MT), and Text-to-Speech (TTS), the system aims to provide low-latency, high-quality translation. Optimized for Apple Silicon (M1/M2/M3) hardware, the system features a modular architecture with a FastAPI backend and a responsive web-based user interface. Key functionalities include real-time transcription, translation, and speech synthesis, dynamic language switching, voice activity detection, and real-time latency visualization. The thesis details the system's architecture, implementation specifics, experimental setup, and performance results, demonstrating its capability to achieve target latencies of 2-3 seconds for standard TTS and 2.5-3.5 seconds for voice cloning. Future enhancements and potential optimizations are also discussed.

## 1. Introduction

### 1.1 Background and Motivation

In an increasingly interconnected world, the demand for effective cross-lingual communication is paramount. Traditional methods of translation, such as human interpreters or delayed text translation, often introduce significant latency and cost, hindering real-time interaction. This is particularly evident in dynamic environments like international conferences, business meetings, or educational settings where immediate understanding is crucial. The advent of advanced artificial intelligence models in speech processing has opened new avenues for bridging these communication gaps, making real-time speech translation a tangible and impactful solution.

### 1.2 Problem Statement

The primary challenge in developing a real-time speech translation system lies in achieving a delicate balance between low latency, high translation quality, and broad accessibility. Existing solutions often compromise on one or more of these aspects. High-quality models can be computationally intensive, leading to unacceptable delays, while faster models may sacrifice accuracy. Furthermore, the integration of multiple complex AI models (STT, MT, TTS) into a cohesive, efficient, and user-friendly pipeline presents significant engineering hurdles, especially when targeting specific hardware optimizations like Apple Silicon.

### 1.3 Goals and Objectives

This project aims to design, implement, and evaluate a real-time live speech translation system with the following objectives:

* **Low-Latency Performance:** Achieve end-to-end translation latency of 2-3 seconds for standard TTS and 2.5-3.5 seconds for voice cloning, suitable for live communication.
* **Modular Architecture:** Develop a robust and scalable system using a client-server architecture with a FastAPI backend and a responsive web-based user interface.
* **State-of-the-Art Open-Source Models:** Integrate and optimize leading open-source models for STT (`faster-whisper`), MT (`CTranslate2` with `Helsinki-NLP Opus-MT`), and TTS (`Piper TTS`, `Coqui TTS` for voice cloning).
* **Hardware Optimization:** Ensure efficient performance on Apple Silicon (M1/M2/M3) hardware.
* **Dynamic Functionality:** Implement dynamic language switching, robust Voice Activity Detection (VAD), and real-time latency visualization within the user interface.
* **User Experience:** Provide a modern, intuitive, and visually appealing web UI that supports both default and thesis-specific content presentation.

### 1.4 Thesis Structure

This thesis is organized into the following chapters:

* **Chapter 1: Introduction** provides an overview of the project, its motivation, the problem it addresses, and its objectives.
* **Chapter 2: Theoretical Background** reviews the fundamental concepts and technologies underpinning speech-to-text, machine translation, and text-to-speech systems.
* **Chapter 3: System Architecture** details the overall design of the real-time speech translation pipeline, including its components and data flow.
* **Chapter 4: Implementation Details** describes the technical implementation of both the frontend and backend components, as well as the integration of the various AI models.
* **Chapter 5: Experimental Setup and Results** presents the methodology for evaluating the system's performance, including latency and quality metrics, and discusses the findings.
* **Chapter 6: Conclusion and Future Work** summarizes the project's achievements, outlines its limitations, and proposes directions for future enhancements.

## 2. Theoretical Background / Literature Review

This chapter provides an overview of the core technologies and theoretical concepts that form the foundation of the real-time speech translation system. We delve into the principles of Speech-to-Text (STT), Machine Translation (MT), Text-to-Speech (TTS), Voice Activity Detection (VAD), and real-time audio processing.

### 2.1 Speech-to-Text (STT)

Speech-to-Text (STT), also known as automatic speech recognition (ASR), is the process of converting spoken language into written text. Modern STT systems primarily rely on deep learning architectures, particularly transformer-based models, which have significantly advanced the state of the art in accuracy and robustness.

* **Evolution of STT:** Briefly discuss early STT systems (Hidden Markov Models, Gaussian Mixture Models) and the paradigm shift brought by deep neural networks (Recurrent Neural Networks, Convolutional Neural Networks).
* **Transformer Models:** Explain the attention mechanism and its role in transformer architectures, which allow for parallel processing and capture long-range dependencies in speech.
* **Whisper and Faster-Whisper:** Detail OpenAI's Whisper model as a robust, general-purpose ASR model trained on a vast dataset. Emphasize `faster-whisper` as an optimized implementation that leverages CTranslate2 for faster inference with reduced memory usage, making it suitable for real-time applications.

#### 2.1.1 Why Faster-Whisper?

The choice of `faster-whisper` as the primary STT engine was driven by several key factors, particularly its suitability for real-time, low-latency applications on diverse hardware, including Apple Silicon.

* **Performance (Speed and Efficiency):** `faster-whisper` is a reimplementation of OpenAI's Whisper model using CTranslate2, a fast inference engine for Transformer models. This optimization significantly reduces inference time and memory usage compared to the original PyTorch implementation of Whisper. For a real-time system, this speed is paramount to minimize the STT component's contribution to overall latency.
* **Accuracy:** OpenAI's Whisper models are renowned for their high accuracy across a wide range of languages and accents, even in noisy environments. `faster-whisper` retains this accuracy while improving performance.
* **Multilingual Support:** Whisper models are trained on a vast dataset of multilingual and multitask supervised data, making them highly effective for language detection and transcription in multiple languages, which is essential for a translation system.
* **Hardware Compatibility:** `faster-whisper` leverages CTranslate2, which is optimized for various hardware, including CPUs and GPUs. This ensures efficient performance on the target Apple Silicon (M1/M2/M3) architecture, as well as broader compatibility for deployment environments.
* **Open-Source and Active Development:** Being open-source, `faster-whisper` benefits from community contributions and ongoing improvements, ensuring its long-term viability and access to the latest optimizations.

**Alternatives Considered:**

* **Original OpenAI Whisper (PyTorch):** While highly accurate, the original PyTorch implementation is significantly slower and more resource-intensive, making it less suitable for real-time streaming applications where low latency is critical.
* **`whisper.cpp`:** This is another highly optimized C++ port of Whisper. It offers excellent performance, especially on CPU. However, `faster-whisper` provides a more direct Python API integration, which aligns better with the FastAPI backend, and its CTranslate2 backend offers competitive performance on Apple Silicon and GPUs.
* **Other ASR APIs (e.g., Google Cloud Speech-to-Text, AWS Transcribe):** Cloud-based APIs offer high accuracy and scalability but introduce network latency, incur costs, and require an internet connection, which might not be ideal for all deployment scenarios or for a bachelor's project focused on local model execution.

The decision to use `faster-whisper` strikes an optimal balance between accuracy, speed, resource efficiency, and ease of integration for this real-time speech translation system.

### 2.2 Machine Translation (MT)

Machine Translation (MT) is the automated process of translating text or speech from one natural language to another. Neural Machine Translation (NMT) has become the dominant approach, outperforming statistical and rule-based methods.

* **Neural Machine Translation (NMT):** Introduce the concept of NMT, typically using encoder-decoder architectures.
* **Transformer-based NMT:** Explain how transformers are applied to MT, enabling high-quality translations by processing entire sequences simultaneously.
* **Opus-MT and SeamlessM4T v2:** Discuss `Helsinki-NLP Opus-MT` models as a collection of pre-trained NMT models covering numerous language pairs, known for their quality and open-source availability. Mention `SeamlessM4T v2` as a more recent, unified model capable of speech-to-speech translation.
* **CTranslate2:** Describe `CTranslate2` as a fast inference engine for NMT models, which optimizes transformer models for CPU and GPU, crucial for achieving low latency in the translation pipeline.

#### 2.2.1 Why CTranslate2 with Opus-MT?

The combination of `CTranslate2` as an inference engine and `Helsinki-NLP Opus-MT` models for machine translation was selected for its balance of performance, quality, and flexibility in a real-time, multi-language environment.

* **Performance (Speed and Efficiency):** `CTranslate2` is a highly optimized C++ and Python library for efficient inference with Transformer models. It provides significant speedups over standard PyTorch or TensorFlow implementations, especially for CPU-bound tasks, which is critical for maintaining low latency in a real-time pipeline. Its optimizations include quantization (e.g., `int8` precision), which reduces model size and speeds up computation without significant loss in accuracy.
* **Broad Language Coverage:** `Helsinki-NLP Opus-MT` is a vast collection of pre-trained neural machine translation models covering hundreds of language pairs. These models are trained on the OPUS corpus, a large collection of parallel corpora, ensuring good quality for many common and less common language combinations. This extensive coverage is vital for a system aiming to support dynamic language switching.
* **Open-Source and Customizable:** Both `CTranslate2` and `Opus-MT` models are open-source, allowing for full control over the translation process, local deployment, and potential future fine-tuning or integration with custom models.
* **Dynamic Model Loading:** The architecture allows for dynamic loading of specific `Opus-MT` models (converted to `CTranslate2` format) based on the user's selected source and target languages, optimizing resource usage by only loading necessary models.

**Alternatives Considered:**

* **Hugging Face Transformers (PyTorch/TensorFlow):** While offering immense flexibility and access to a vast array of NMT models, running inference directly with the full `transformers` library can be slower and more memory-intensive than `CTranslate2` for production-grade real-time applications, especially without dedicated GPU acceleration.
* **Cloud MT APIs (e.g., Google Translate API, DeepL API):** These services offer high-quality translations and scalability. However, they introduce network latency, incur costs per translation, and require continuous internet connectivity. For a bachelor's project focused on local model execution and minimizing external dependencies, a self-hosted solution was preferred.
* **SeamlessM4T v2:** While a promising unified speech-to-speech translation model, its complexity and resource requirements were deemed too high for the initial scope of a bachelor's project, which prioritizes a modular pipeline with distinct STT, MT, and TTS components for clearer evaluation and optimization.

The choice of `CTranslate2` with `Opus-MT` provides a robust, efficient, and flexible machine translation solution that aligns perfectly with the real-time and performance objectives of this project.

### 2.3 Text-to-Speech (TTS)

Text-to-Speech (TTS) synthesis converts written text into spoken audio. Recent advancements in deep learning have led to highly natural and expressive synthetic voices.

* **TTS Architectures:** Briefly cover concatenative, parametric, and neural TTS approaches. Focus on neural TTS models that generate speech directly from text using end-to-end deep learning.
* **Piper TTS:** Detail `Piper TTS` as an efficient and high-quality open-source TTS system, known for its small model sizes and fast inference, making it ideal for real-time applications.
* **Coqui TTS (XTTS v2) for Voice Cloning:** Explain the concept of voice cloning, where a TTS model can synthesize speech in a target speaker's voice from a short audio sample. Describe `Coqui TTS` with the `XTTS v2` model as a state-of-the-art system capable of zero-shot, cross-lingual voice cloning, adding a personalized dimension to the translation output.

#### 2.3.1 Why Piper TTS and Coqui TTS?

The selection of `Piper TTS` and `Coqui TTS (XTTS v2)` provides a dual-pronged approach to text-to-speech synthesis, offering both fast, generic speech and high-quality voice-cloned output, catering to different user preferences and performance requirements.

**Piper TTS:**

* **Speed and Efficiency:** `Piper TTS` is highly optimized for fast, local inference, making it exceptionally suitable for real-time applications where low latency is critical. Its small model sizes contribute to quick loading and minimal resource consumption.
* **High Quality (Generic Voice):** Despite its efficiency, Piper produces natural-sounding speech with good clarity, making it an excellent choice for a default, fast TTS option.
* **Open-Source and Offline Capability:** Being open-source and designed for local execution, Piper allows for offline operation and full control over the synthesis process, aligning with the project's goals of a standalone system.
* **Multilingual Support:** Piper supports a growing number of languages, which is important for a translation system.

**Coqui TTS (XTTS v2) for Voice Cloning:**

* **Zero-Shot Voice Cloning:** `Coqui TTS` with the `XTTS v2` model enables zero-shot voice cloning, meaning it can synthesize speech in a target speaker's voice from just a few seconds of audio, without requiring extensive training. This feature allows the system to synthesize translated speech in the original speaker's voice, providing a highly personalized and immersive communication experience.
* **Cross-Lingual Transfer:** XTTS v2 is specifically designed for cross-lingual voice cloning, meaning a voice recorded in one language (e.g., English) can be used to synthesize speech in another language (e.g., Czech), which is crucial for a translation application. This is a key differentiator from many other TTS systems.
* **Official Multilingual Support:** XTTS v2 officially supports 17 languages, including English and Czech, ensuring high-quality synthesis for the project's target language pairs.
* **Higher Quality (Expressiveness):** Voice cloning models often aim for higher expressiveness and naturalness, capturing not just the timbre but also the prosody and emotional characteristics of the speaker's voice.
* **Streaming Capability:** XTTS v2 supports streaming synthesis, allowing for reduced perceived latency by generating and playing audio incrementally rather than waiting for the entire text to be synthesized.

**Trade-offs and Decision:**

The decision to include both Piper and Coqui TTS is a strategic one:

* **Piper** serves as the default, high-performance option for scenarios where speed and efficiency are paramount (RTF ~0.05, latency ~0.2-0.3s), and a generic voice is acceptable.
* **Coqui TTS** provides the advanced voice cloning feature, offering a premium, personalized experience, albeit with higher computational demands and increased latency (RTF ~1.72 on CPU, perceived latency ~2-3s with streaming) compared to Piper.

This dual approach allows the system to be flexible and cater to different user needs, while also showcasing advanced TTS capabilities within the project.

**Why Coqui TTS over Coqui TTS:**

During development, `Coqui TTS` was initially considered for voice cloning. However, `Coqui TTS (XTTS v2)` was ultimately selected for the following reasons:

* **Stability:** Coqui TTS exhibited instability issues on Apple Silicon (MPS backend crashes), requiring frequent fallbacks to CPU. Coqui TTS demonstrated more robust and consistent performance across different hardware configurations.
* **Official Multilingual Support:** While Coqui TTS showed promise, XTTS v2 provides official, well-documented support for 17 languages, including Czech, which is critical for this project. Coqui TTS's multilingual capabilities were less mature at the time of development.
* **Better Documentation and Community:** Coqui TTS benefits from comprehensive documentation, active community support, and regular updates, making it easier to integrate, troubleshoot, and optimize.
* **Cross-Lingual Voice Cloning:** XTTS v2 is specifically designed for cross-lingual scenarios, a core requirement for this translation system, whereas Coqui TTS's cross-lingual capabilities required more experimental tuning.

**Alternatives Considered:**

* **Coqui TTS:** As mentioned, Coqui TTS was initially explored but ultimately replaced by Coqui TTS due to stability and multilingual support concerns.
* **Other Open-Source TTS (e.g., ESPnet, StyleTTS):** While powerful, these frameworks can be more complex to integrate and might not offer the same out-of-the-box real-time performance or specific zero-shot cross-lingual voice cloning capabilities as efficiently as Coqui TTS for the project's scope.
* **Cloud TTS APIs (e.g., Google Cloud Text-to-Speech, Amazon Polly):** Similar to MT APIs, cloud TTS services offer high quality and a wide range of voices, but introduce network latency, costs, and dependency on external services, which are less desirable for a local, open-source focused project.

The combination of Piper and Coqui TTS provides a comprehensive and optimized TTS solution for the real-time speech translation system.

### 2.4 Voice Activity Detection (VAD)

Voice Activity Detection (VAD) is a technique used to detect the presence or absence of human speech in an audio stream. It is critical for real-time speech processing pipelines to efficiently segment audio, reduce computational load, and improve the accuracy of downstream STT and MT models.

* **Importance of VAD:** Explain how VAD helps in distinguishing speech from silence or background noise, enabling the system to process only relevant audio segments.
* **WebRTC VAD:** Describe `webrtcvad` as a robust and widely used VAD algorithm, known for its efficiency and configurable aggressiveness levels, which allow for tuning its sensitivity to speech.

#### 2.4.1 Why WebRTC VAD?

The `webrtcvad` library was chosen for Voice Activity Detection due to its proven robustness, efficiency, and suitability for real-time audio processing, particularly in the context of WebRTC-based applications.

* **Robustness and Accuracy:** `webrtcvad` is a highly reliable VAD algorithm developed by Google for use in WebRTC. It is designed to accurately distinguish human speech from background noise and silence, even in challenging audio environments. This robustness is crucial for ensuring that only relevant audio segments are passed to the computationally intensive STT and MT models.
* **Efficiency and Low Latency:** The algorithm is lightweight and optimized for real-time performance. It operates on small audio frames (e.g., 10, 20, or 30 ms), allowing for very low-latency speech detection. This efficiency directly contributes to minimizing the overall end-to-end latency of the translation pipeline.
* **Configurable Aggressiveness:** `webrtcvad` offers configurable aggressiveness levels (0-3), allowing for fine-tuning its sensitivity to speech. This flexibility is important for adapting the VAD to different use cases or noise environments, balancing between detecting all speech and minimizing false positives from noise.
* **Integration with WebRTC Standard:** Given that the frontend uses WebRTC for audio capture, using `webrtcvad` aligns well with the overall architecture and leverages a standard component from the WebRTC ecosystem.
* **Open-Source and Widely Used:** Its open-source nature and widespread adoption ensure good documentation, community support, and continuous improvement.

**Alternatives Considered:**

* **STT Model's Internal VAD:** Some STT models (like Faster-Whisper) include their own VAD filters. While convenient, relying solely on the STT model's internal VAD might not offer the same level of fine-grained control or real-time responsiveness as a dedicated VAD module. Using a separate `webrtcvad` allows for pre-filtering audio before it reaches the STT model, further optimizing the pipeline.
* **Deep Learning-based VAD Models:** More advanced VAD models based on deep learning (e.g., Silero VAD) can offer higher accuracy in very complex scenarios. However, they typically have larger model sizes and higher computational requirements, which could introduce additional latency and resource overhead, making them less suitable for the low-latency constraints of this project. `webrtcvad` provides an excellent balance for this application.
* **Simple Thresholding:** Basic VAD based on audio volume (RMS) thresholding is simple to implement but highly susceptible to noise and varying speaker volumes, leading to poor performance in real-world scenarios. The project does use a pre-VAD RMS threshold to filter out very low-level noise before engaging the more robust `webrtcvad`.

The integration of `webrtcvad` ensures efficient and accurate speech detection, which is a cornerstone for the real-time performance and resource optimization of the entire speech translation system.

### 2.5 Real-time Audio Processing and Streaming

Achieving real-time performance in a complex pipeline requires careful consideration of audio processing and data streaming techniques. This section details the strategies employed to ensure low latency and efficient data flow.

* **Audio Chunking:** Explain the concept of dividing continuous audio streams into smaller, manageable chunks for incremental processing. This is fundamental for real-time systems, as it allows for processing to begin before an entire utterance is complete, significantly reducing perceived latency. The system uses fixed-size audio frames (e.g., 30ms for VAD) and larger streaming chunks (e.g., 500ms for STT/MT/TTS) to balance responsiveness with model efficiency.
* **WebSockets:** Describe WebSockets as a full-duplex communication protocol over a single TCP connection, ideal for real-time, bidirectional data exchange between the frontend and backend. WebSockets provide a persistent connection, eliminating the overhead of repeated HTTP requests and enabling efficient streaming of raw audio bytes from the client to the server, and processed text/audio back to the client.
* **Latency Considerations:** Discuss the various sources of latency in the pipeline (audio capture, VAD, STT, MT, TTS inference, network transmission) and strategies to minimize them. This includes:
  * **Frontend Audio Buffering:** Minimizing client-side audio buffering before sending to the backend.
  * **Asynchronous Backend Processing:** Utilizing FastAPI's asynchronous capabilities and offloading CPU-bound tasks to a thread pool (`loop.run_in_executor`) to prevent the main event loop from blocking.
  * **Optimized Model Inference:** Leveraging `faster-whisper` and `CTranslate2` for their highly optimized inference engines.
  * **Efficient Data Serialization:** Using efficient methods for encoding and decoding audio and text data over WebSockets.

#### 2.5.1 Why WebSockets and Audio Chunking?

The choice of WebSockets for communication and audio chunking for processing is central to achieving the low-latency, real-time requirements of the speech translation system.

* **Real-time Bidirectional Communication (WebSockets):**
  * **Low Overhead:** Unlike traditional HTTP requests, WebSockets establish a persistent connection, drastically reducing the overhead associated with connection setup and teardown for each message. This is crucial for continuous, low-latency data streams.
  * **Full-Duplex:** WebSockets allow for simultaneous sending and receiving of data, which is essential for a real-time translation system where audio is streamed to the backend while transcriptions, translations, and synthesized audio are streamed back to the frontend concurrently.
  * **Efficiency:** They are designed for efficient handling of small, frequent messages, making them ideal for streaming audio frames and receiving incremental results.
* **Reduced Latency (Audio Chunking):**
  * **Incremental Processing:** By dividing the continuous audio input into small chunks, the system can start processing and generating output before the speaker has finished their utterance. This significantly reduces the perceived end-to-end latency, making the interaction feel more natural.
  * **Resource Management:** Processing smaller chunks allows for more efficient memory management and prevents large audio buffers from accumulating, which could lead to increased latency and resource consumption.
  * **Responsiveness:** Smaller chunks enable faster feedback loops, allowing the system to react more quickly to changes in speech (e.g., detecting silence, switching languages).

**Alternatives Considered:**

* **HTTP/REST APIs:** While suitable for request-response patterns, HTTP APIs introduce significant overhead for continuous streaming due to connection setup/teardown and header transmission for each chunk. This would result in unacceptably high latency for a real-time system.
* **Server-Sent Events (SSE):** SSE allows for one-way streaming from server to client. While useful for real-time updates, it does not support bidirectional communication, making it unsuitable for sending audio input from the client to the server.
* **Long Polling:** This technique simulates real-time communication over HTTP but is less efficient and more complex to manage than WebSockets for truly bidirectional, high-frequency data exchange.

The combination of WebSockets and intelligent audio chunking provides the necessary foundation for a highly responsive and efficient real-time speech translation pipeline.

### 2.6 Comparison of Key Models

| Component | Model/Technology                 | Key Features                                         | Advantages for Real-time                                                      |
| :-------- | :------------------------------- | :--------------------------------------------------- | :---------------------------------------------------------------------------- |
| STT       | `faster-whisper`               | Transformer-based, multilingual, optimized inference | High accuracy, significantly faster than original Whisper, low resource usage |
| MT        | `CTranslate2` with `Opus-MT` | Efficient inference engine, broad language coverage  | Fast and efficient translation, optimized for various hardware                |
| TTS       | `Piper TTS`                    | Small model size, high quality, fast inference       | Very low latency, natural-sounding speech, suitable for embedded systems      |
| TTS       | `Coqui TTS` (or `XTTS v2`)   | Voice cloning, expressive speech                     | Personalized speech output, real-time cloning capabilities                    |
| VAD       | `webrtcvad`                    | Robust, configurable aggressiveness                  | Efficient speech detection, reduces processing of silence                     |

## 3. System Architecture

This chapter details the overall design and architecture of the real-time speech translation system. It outlines the client-server model, the detailed pipeline flow, and the individual components that work in concert to achieve seamless cross-lingual communication.

### 3.1 Overall System Design

The system employs a modular client-server architecture to ensure scalability, maintainability, and efficient resource utilization. The frontend, a web-based user interface, handles audio capture and presentation, while the backend, powered by FastAPI, orchestrates the complex speech processing and translation tasks.

#### 3.1.1 Why a Client-Server Architecture with FastAPI and WebRTC?

The choice of a client-server architecture, specifically utilizing a FastAPI backend and a WebRTC-enabled web frontend, is fundamental to meeting the project's objectives of real-time, scalable, and cross-platform speech translation.

* **Separation of Concerns:**
  * **Frontend (UI):** The web-based frontend (`ui/`) is responsible for user interaction, microphone access, displaying real-time text, and playing back translated audio. This keeps the user interface responsive and platform-agnostic (runs in any modern browser).
  * **Backend (Processing):** The FastAPI backend (`backend/`) is dedicated to the computationally intensive tasks of STT, MT, and TTS. This allows for centralized management of AI models, efficient resource allocation, and potential for future scaling (e.g., GPU acceleration, microservices).
* **Real-time Audio Streaming (WebRTC & WebSockets):**
  * **WebRTC for Capture:** `navigator.mediaDevices.getUserMedia` (part of WebRTC) is the standard browser API for accessing the user's microphone. It provides low-latency, high-quality audio capture directly in the browser, eliminating the need for proprietary drivers or plugins on the client side.
  * **WebSockets for Backend Communication:** As discussed in Section 2.5.1, WebSockets provide a persistent, full-duplex channel for efficient, low-latency streaming of audio data to the backend and real-time results back to the frontend. This is crucial for the continuous flow of a live translation system.
* **High-Performance Backend (FastAPI):**
  * **Asynchronous Capabilities:** FastAPI is built on Starlette and Pydantic, offering excellent asynchronous support. This allows the backend to handle multiple concurrent WebSocket connections (users) efficiently without blocking the main event loop, which is critical for scalability.
  * **Performance:** FastAPI is one of the fastest Python web frameworks, making it well-suited for a real-time application where every millisecond counts.
  * **Developer Experience:** Its modern Python features, automatic data validation, and interactive API documentation (Swagger UI) streamline development and maintenance.
* **Cross-Platform Accessibility (Web-based UI):**
  * By being a web application, the frontend is inherently cross-platform, accessible from any device with a modern web browser (desktop, laptop, tablet). This significantly broadens the reach of the application without requiring platform-specific client installations.
* **Scalability:** The client-server model naturally supports horizontal scaling of the backend. As user load increases, more FastAPI instances can be deployed and managed (e.g., with Docker and Kubernetes in future phases) to distribute the processing workload.

**Alternatives Considered:**

* **Desktop Application:** A native desktop application could offer tighter OS integration (e.g., easier virtual audio device setup) but would sacrifice cross-platform accessibility and require separate development and maintenance for each operating system (macOS, Windows, Linux).
* **Pure Frontend (Browser-only processing):** While some smaller models can run in the browser (e.g., WebAssembly versions of Whisper), a full STT-MT-TTS pipeline with voice cloning, especially with larger models, would be too computationally intensive for most client-side browsers, leading to poor performance and high battery consumption.
* **Other Backend Frameworks (e.g., Flask, Django):** While viable, FastAPI's native asynchronous support and performance characteristics make it a superior choice for high-throughput, real-time WebSocket applications compared to traditional synchronous frameworks like Flask or Django (without extensive asynchronous extensions).

The chosen architecture provides a robust, scalable, and performant foundation for the real-time speech translation system, balancing development efficiency with critical performance requirements.

### 3.2 Detailed Pipeline Flow

The real-time speech translation pipeline is a sequential process where audio data flows through several stages, each performing a specific task. The entire process is optimized for low latency to provide a near-instantaneous translation experience.

```mermaid
graph LR
    A[Microphone Input] --> B{WebRTC VAD}
    B -- Speech --> C[Faster-Whisper STT]
    B -- Silence --> A
    C --> D[CTranslate2 MT<br/>EN↔SK]
    D --> E{TTS Model}
    E -- Fast --> F[Piper TTS]
    E -- Cloned --> G[Coqui TTS XTTS v2 + Voice.wav]
    F --> H[Audio Output]
    G --> H
    H --> I[Virtual Audio Device<br/>(BlackHole/VB-Cable)]
    style E fill:#f9f,stroke:#333
```

1. **Microphone Input:** Audio is captured from the user's microphone in the browser using WebRTC (`navigator.mediaDevices.getUserMedia`).
2. **WebRTC VAD:** The captured audio stream is fed into the WebRTC Voice Activity Detection module, which intelligently identifies segments containing human speech and filters out silence or background noise.
3. **Faster-Whisper STT:** Speech segments are sent to the `faster-whisper` Speech-to-Text model, which transcribes the spoken words into text in the source language.
4. **CTranslate2 MT:** The transcribed text is then passed to the `CTranslate2` Machine Translation engine, which uses `Helsinki-NLP Opus-MT` models to translate the text into the target language.
5. **TTS Model Selection:** Based on user preference, either `Piper TTS` for fast, standard speech synthesis or `Coqui TTS` for voice-cloned speech synthesis is selected.
6. **Piper TTS / Coqui TTS:** The translated text is synthesized into audio by the chosen TTS model. If `Coqui TTS` is used, a pre-recorded or uploaded speaker voice profile (`Voice.wav`) is utilized for voice cloning.
7. **Audio Output:** The synthesized audio is streamed back to the frontend.
8. **Virtual Audio Device (BlackHole/VB-Cable):** The frontend plays the synthesized audio through the browser's Web Audio API, routing it to a virtual audio device (e.g., BlackHole on macOS, VB-Cable on Windows). This virtual device then acts as a microphone input for other conferencing applications (e.g., Google Meet, Zoom), allowing participants to hear the translated speech. The user initiating the translation will typically not hear their own translated output directly from their speakers.

### 3.3 Component Breakdown

The system is composed of several interconnected modules, each specialized for a particular function.

* **Frontend (UI):**
  * `ui/home/home.html`: Main HTML structure, including dual-mode layout (default/thesis), header, footer, and content sections.
  * `ui/home/home.css`: Styling for home page elements, navigation, and mode-specific adjustments.
  * `ui/home/home.js`: JavaScript for dynamic UI behavior, header scroll, hamburger menu, theme switching, mode switching, bubble navigation, Chart.js initialization, and code snippet copy functionality.
  * `ui/audio-processor.js`: Web Audio Worklet for efficient, off-main-thread audio processing.
  * `ui/global-styles.css`, `styles.css`: Global CSS for consistent theming and distinct section coloring.
* **Backend (FastAPI):**
  * `app.py`, `backend/main.py`: Main FastAPI application, WebSocket handling, and orchestration logic.
  * `backend/stt/faster_whisper_stt.py`: Wrapper for the `faster-whisper` STT model.
  * `backend/mt/ctranslate2_mt.py`: Wrapper for `CTranslate2`-based Machine Translation.
  * `backend/tts/piper_tts.py`: Wrapper for `Piper TTS` models.
  * `backend/tts/f5_tts.py`: Wrapper for `Coqui TTS` (or `XTTS v2`) for voice cloning.
  * `backend/utils/audio_utils.py`: Utility functions for audio manipulation.
  * `backend/utils/auth.py`, `backend/utils/db_manager.py`: Authentication and database management (if applicable).
* **Models and Data:**
  * `backend/tts/piper_models/`: Directory for downloaded Piper TTS models.
  * `ct2_models/`: Directory for converted CTranslate2 MT models.
  * `speaker_voices/`: Directory for stored speaker voice profiles for cloning.
  * `certs/`: SSL certificates for HTTPS/WSS communication.

### 3.4 Key Technologies

| Category                 | Technology/Tool                | Role in System                                                      |
| :----------------------- | :----------------------------- | :------------------------------------------------------------------ |
| **Frontend**       | HTML, CSS, JavaScript          | Core web technologies for user interface and interactivity          |
|                          | Web Audio API                  | Browser-based audio capture and processing                          |
|                          | WebSockets                     | Real-time, bidirectional communication between frontend and backend |
|                          | Chart.js                       | Interactive data visualization for latency metrics                  |
| **Backend**        | Python                         | Primary programming language                                        |
|                          | FastAPI                        | High-performance web framework for API and WebSocket endpoints      |
|                          | `faster-whisper`             | Optimized Speech-to-Text (STT) model                                |
|                          | `CTranslate2`                | Fast inference engine for Machine Translation (MT) models           |
|                          | `Helsinki-NLP Opus-MT`       | Pre-trained Neural Machine Translation models                       |
|                          | `Piper TTS`                  | Efficient and high-quality Text-to-Speech (TTS) model               |
|                          | `Coqui TTS` (or `XTTS v2`) | Real-time voice cloning Text-to-Speech (TTS) model                  |
|                          | `webrtcvad`                  | Voice Activity Detection (VAD) library                              |
| **Deployment/Dev** | Apple Silicon (M1/M2/M3)       | Target hardware for optimized performance                           |
|                          | `ffmpeg`                     | Audio processing utility                                            |
|                          | `openssl`                    | For generating SSL certificates                                     |

#### 3.4.1 Why These Key Technologies?

The selection of specific technologies and tools across frontend, backend, and deployment categories was made to ensure the system is robust, performant, scalable, and maintainable, while leveraging open-source solutions suitable for a bachelor's project.

* **HTML, CSS, JavaScript (Frontend Core):**
  * **Why:** These are the foundational technologies for web development, offering universal browser compatibility and flexibility. For a bachelor's project, they provide a direct and understandable way to build a responsive UI without the added complexity of a heavy frontend framework (like React, Angular, Vue), which would introduce a steeper learning curve and additional build steps.
  * **Alternatives Considered:** Modern JavaScript frameworks (React, Vue, Angular) offer component-based development and state management, but were deemed overkill for the current UI complexity and would divert focus from the core AI pipeline.
* **Web Audio API (Frontend Audio Processing):**
  * **Why:** Provides powerful, low-level control over audio processing directly in the browser. It enables efficient microphone capture, real-time manipulation (e.g., resampling, chunking via Web Audio Worklets), and playback of synthesized audio without relying on server-side processing for these client-specific tasks. This is crucial for minimizing frontend-induced latency.
  * **Alternatives Considered:** Simpler `MediaRecorder` API is easier to use but offers less control over real-time audio manipulation and direct access to raw audio buffers for advanced processing.
* **WebSockets (Real-time Communication):**
  * **Why:** Essential for the bidirectional, low-latency streaming of audio and text data between the frontend and backend. As detailed in Section 2.5.1, WebSockets significantly outperform traditional HTTP for continuous real-time communication by reducing overhead.
  * **Alternatives Considered:** HTTP polling or Server-Sent Events (SSE) were considered but rejected due to higher latency (polling) or lack of bidirectional support (SSE).
* **Chart.js (Latency Visualization):**
  * **Why:** A lightweight, flexible JavaScript charting library that is easy to integrate into a plain HTML/CSS/JS frontend. It allows for clear and interactive visualization of real-time latency metrics, which is a key objective for evaluating the system's performance.
  * **Alternatives Considered:** D3.js offers more power but has a steeper learning curve. Other charting libraries might be heavier or less straightforward to integrate without a framework.
* **Python (Backend Language):**
  * **Why:** Python is the de facto language for AI/ML development, with a rich ecosystem of libraries (PyTorch, NumPy, Hugging Face, CTranslate2, Piper, Coqui TTS) that are directly used by the backend models. Its readability and extensive community support accelerate development.
  * **Alternatives Considered:** Languages like C++ or Go could offer higher raw performance but lack the extensive and mature AI/ML ecosystem of Python, making model integration significantly more complex.
* **FastAPI (Backend Web Framework):**
  * **Why:** Chosen for its exceptional performance, native asynchronous support, and modern developer experience. It efficiently handles concurrent WebSocket connections and integrates seamlessly with Python's `asyncio` for non-blocking I/O and CPU-bound task offloading. Its automatic data validation and documentation features are also highly beneficial.
  * **Alternatives Considered:** Flask is a popular lightweight framework but requires extensions for robust asynchronous handling. Django is a full-stack framework, often overkill for API-centric applications. Node.js with Express could offer good async performance but would require a different language stack for the AI models.
* **Apple Silicon (M1/M2/M3) (Target Hardware):**
  * **Why:** The development environment leverages Apple Silicon for its high performance-per-watt, integrated GPU (MPS), and unified memory architecture. This provides a powerful local development and testing platform for AI models, allowing for efficient iteration and optimization.
  * **Alternatives Considered:** Standard x86 CPUs or dedicated NVIDIA GPUs (CUDA) are common for AI. While CUDA will be considered for deployment, Apple Silicon offers a strong local development experience.
* **`ffmpeg` (Audio Processing Utility):**
  * **Why:** A powerful, industry-standard command-line tool for handling multimedia data. It is used for robust audio format conversions (e.g., WebM to WAV) and ensuring consistent audio properties (sample rate, channels) before processing by AI models.
  * **Alternatives Considered:** Pure Python audio libraries might lack the comprehensive format support or efficiency of `ffmpeg` for complex conversions.
* **`openssl` (SSL Certificates):**
  * **Why:** Used for generating self-signed SSL certificates, enabling HTTPS/WSS communication for secure local development and testing. This is a standard practice for securing web applications.
  * **Alternatives Considered:** Commercial SSL certificates are for production environments. Other tools exist for self-signed certs, but `openssl` is widely available and robust.

This carefully selected technology stack provides a solid foundation for building and evaluating the real-time speech translation system, addressing both functional and non-functional requirements effectively.

## 4. Implementation Details

This chapter delves into the technical implementation of the real-time speech translation system, detailing the development of both the frontend and backend components, as well as the integration and optimization of the various AI models.

### 4.1 Frontend Implementation

The frontend is a modern web application built with standard web technologies (HTML, CSS, JavaScript) to ensure broad compatibility and a rich user experience, following the UI/UX design guidance inspired by "Lingonberry Mail".

* **HTML Structure (`ui/home/home.html`):** The main HTML file defines the dual-mode layout, including the header, footer, and distinct content sections for "Default Mode" and "Thesis Mode." It incorporates placeholders for dynamic content such as interactive charts (using `<canvas>` elements for Chart.js) and data tables.
* **CSS Styling (`ui/home/home.css`, `ui/global-styles.css`, `styles.css`):**
  * `ui/global-styles.css` and `styles.css` define global CSS variables for consistent theming (day/night mode), typography, and base element styling. The color palette is adapted from "Lingonberry Mail" with specific variables for light and dark themes (e.g., `--primary-color`, `--background-color-start`, `--text-color`).
  * `ui/home/home.css` provides specific styles for the home page, including responsive design for the header, navigation (hamburger menu, bubble navigation), and distinct visual treatments for each section in both default and thesis modes.
  * **Layout & Structure Principles:**
    * **Full-Screen Sections:** Each major section (Hero, About, Features, How-To, Demo, About, FAQ) occupies `min-h-screen` and is designed as a distinct "slide" with smooth scrolling behavior. Content within sections is centered horizontally using responsive containers.
    * **"Floating Container" for Content:** Content within sections resides in a "floating container" (e.g., `div` with `bg-surface-color`, `p-8`, `rounded-xl`, `shadow-lg`, `border border-border-color`) for visual separation.
    * **Header:** Fixed at the top (`fixed w-full top-0 z-40`), with dynamic visibility (hides on scroll down, reappears on scroll up in default mode). Uses `font-montserrat` for navigation links and buttons, and a theme toggle icon.
    * **Footer:** A distinct section at the bottom (`p-8 text-center text-text-color font-montserrat`) with copyright information.
    * **Floating Sidebar Navigation:** Small, circular "bubbles" (`w-5 h-5 rounded-full`) fixed on the left side (`fixed left-8 top-1/2 -translate-y-1/2 z-30`). Hidden in the Hero section, visible otherwise, with smooth transitions and active states.
  * Custom styles are applied for elements like `.hero-illustration`, `.feature-icon`, `.step-icon`, `.section-subtitle`, `.thesis-figure`, `.thesis-table-container`, `.data-table`, and `.code-snippet-container` to achieve the desired aesthetic and Notion-like code block styling.
* **JavaScript Logic (`ui/home/home.js`, `ui/audio-processor.js`, `ui/theme-toggle.js`):**
  * `ui/home/home.js` manages core UI interactivity:
    * **Header Behavior:** Implements logic for the header to hide on scroll down and reappear on scroll up in default mode, and to be completely hidden in thesis mode.
    * **Hamburger Menu:** Controls the opening and closing of the mobile navigation menu.
    * **Theme-Dependent Images:** Dynamically switches signature images in the footer based on the active theme.
    * **Mode Switching:** Handles the logic for toggling between "Default Mode" and "Thesis Mode," including content visibility and smooth scrolling to the top.
    * **Bubble Navigation:** Manages active states and smooth scrolling for the `section-nav` elements.
    * **Chart.js Integration:** Initializes and updates interactive charts (e.g., `architectureChart`, `performanceChart`) for visualizing system metrics in thesis mode.
    * **Code Snippet Copy Functionality:** Provides a `copyCode` function to allow users to easily copy code blocks to their clipboard.
  * `ui/audio-processor.js` implements a Web Audio Worklet for efficient, off-main-thread audio processing. This ensures that audio capture and preprocessing (e.g., resampling, chunking) do not block the main UI thread, contributing to low latency.
  * `ui/theme-toggle.js` handles the day/night mode switching logic, persisting user preferences.
* **Web Audio API and WebSockets:** The frontend utilizes the Web Audio API to access the user's microphone, process audio streams, and send raw audio data in chunks to the backend via WebSockets. WebSockets also facilitate real-time reception of transcriptions, translations, and synthesized audio from the backend.

### 4.2 Backend Implementation

The backend is built with Python using the FastAPI framework, chosen for its high performance, asynchronous capabilities, and ease of developing API and WebSocket endpoints.

* **FastAPI Application (`app.py`, `backend/main.py`):**
  * `backend/main.py` defines the main FastAPI application, including routes for health checks, model initialization, and WebSocket endpoints for real-time audio streaming.
  * It orchestrates the entire pipeline, managing the flow of audio data through VAD, STT, MT, and TTS modules.
  * Configuration parameters (e.g., `STT_MODEL_SIZE`, `VAD_AGGRESSIVENESS`, `PIPER_MODEL_MAPPING`) are defined here, allowing for flexible system tuning.
* **WebSocket Handling:** FastAPI's WebSocket capabilities are used to establish a persistent, full-duplex connection with the frontend. This enables efficient, low-latency streaming of audio input and real-time delivery of processed text and audio output.
* **Model Orchestration:** The backend manages the loading and inference of all AI models. It dynamically loads MT and TTS models based on the selected language pairs and TTS engine.
* **SSL Certificates:** Self-signed SSL certificates (generated via `openssl`) are used to enable HTTPS/WSS communication, ensuring secure data transfer.

### 4.3 Model Integration and Optimization

Each AI model is integrated through dedicated Python wrappers, ensuring a consistent interface and allowing for specific optimizations.

* **Speech-to-Text (`backend/stt/faster_whisper_stt.py`):**
  * The `FasterWhisperSTT` class wraps the `faster-whisper` library.
  * It loads the specified Whisper model (e.g., `large-v3`) and utilizes `CTranslate2` for optimized inference on Apple Silicon, significantly reducing transcription latency compared to the original Whisper implementation.
  * The wrapper handles audio input, performs transcription, and returns the text.
* **Machine Translation (`backend/mt/ctranslate2_mt.py`):**
  * The `CTranslate2MT` class manages the loading and inference of `Helsinki-NLP Opus-MT` models converted to `CTranslate2` format.
  * It supports dynamic loading of different language pair models (e.g., `en-sk`, `sk-en`) based on user selection.
  * `CTranslate2`'s optimizations ensure fast and efficient text translation.
* **Text-to-Speech (`backend/tts/piper_tts.py`, `backend/tts/coqui_tts.py`):**
  * `PiperTTS` class wraps the `Piper TTS` library, providing an interface for fast, natural-sounding speech synthesis. Models are stored in `backend/tts/piper_models/`.
  * `CoquiTTS` class handles real-time voice cloning using the `XTTS v2` model from Coqui TTS. It takes a speaker's audio sample (`Voice.wav` from `speaker_voices/`) and synthesizes the translated text in that voice. This involves loading the XTTS v2 model and managing speaker embeddings.
  * **Speaker Embedding Caching:** To improve performance, the `CoquiTTS` wrapper implements speaker embedding caching. When a voice is first used, the speaker embedding is computed and cached in memory. Subsequent syntheses with the same voice reuse the cached embedding, reducing latency by approximately 1.9%.
  * **Hybrid Synthesis Strategy:** The Coqui TTS implementation uses a hybrid approach to balance quality and latency:
    * **Short texts (<200 chars):** Single-shot synthesis is used to generate the entire audio at once, eliminating chunk boundary artifacts.
    * **Long texts (≥200 chars):** Sentence-level streaming synthesis is used to reduce perceived latency by generating and playing audio incrementally.
  * Both wrappers are designed for low-latency audio generation.
* **Voice Activity Detection (`webrtcvad`):**
  * Integrated directly into the backend's audio processing pipeline.
  * `webrtcvad` efficiently identifies speech segments, preventing silent audio from being sent to STT and MT models, thereby reducing computational load and improving overall pipeline efficiency. Its aggressiveness level is configurable.

### 4.4 Data Management

* **Model Storage:** Pre-trained models for Piper TTS and CTranslate2 MT are stored locally in `backend/tts/piper_models/` and `ct2_models/` respectively, allowing for offline operation and faster loading times after initial download.
* **Speaker Voice Profiles:** User-recorded or uploaded voice samples for Coqui TTS voice cloning are stored in the `speaker_voices/` directory, managed by the backend.
* **Configuration:** Key system parameters are managed within `backend/main.py` and can be adjusted to fine-tune performance or behavior.

### 4.5 Code Snippets

#### Frontend WebSocket Connection (Simplified `ui/home/home.js`)

```javascript
// Example: Simplified WebSocket connection and message handling
let ws;
function connectWebSocket() {
    ws = new WebSocket("wss://localhost:8000/ws");

    ws.onopen = () => {
        console.log("WebSocket connection established.");
        // Send initial configuration or start audio stream
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "transcription") {
            document.getElementById("transcription-output").innerText = data.text;
        } else if (data.type === "translation") {
            document.getElementById("translation-output").innerText = data.text;
        } else if (data.type === "audio") {
            // Play synthesized audio
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const audioBuffer = audioContext.decodeAudioData(base64ToArrayBuffer(data.audio));
            audioBuffer.then(buffer => {
                const source = audioContext.createBufferSource();
                source.buffer = buffer;
                source.connect(audioContext.destination);
                source.start(0);
            });
        }
    };

    ws.onclose = () => {
        console.log("WebSocket connection closed.");
        // Attempt to reconnect or show error
    };

    ws.onerror = (error) => {
        console.error("WebSocket error:", error);
    };
}

// Helper to convert base64 to ArrayBuffer (for audio playback)
function base64ToArrayBuffer(base64) {
    const binaryString = window.atob(base64);
    const len = binaryString.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
}
```

#### Backend WebSocket Endpoint (Simplified `backend/main.py`)

```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from backend.stt.faster_whisper_stt import FasterWhisperSTT
from backend.mt.ctranslate2_mt import CTranslate2MT
from backend.tts.piper_tts import PiperTTS
from backend.tts.f5_tts import F5TTS
from backend.utils.audio_utils import resample_audio, bytes_to_float_array
import webrtcvad
import asyncio
import json
import base64

app = FastAPI()

# Initialize models (simplified for example)
stt_model = FasterWhisperSTT(model_size="small")
mt_model = CTranslate2MT(model_name="Helsinki-NLP/opus-mt-en-sk")
piper_tts = PiperTTS(model_id="en_US-ryan-medium")
f5_tts = F5TTS() # Requires speaker_wav_path to be set

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected.")
    vad = webrtcvad.Vad(3) # Aggressiveness mode 3 (most aggressive)
    audio_buffer = []
    speech_detected = False

    try:
        while True:
            data = await websocket.receive_bytes()
            float_array = bytes_to_float_array(data)
            # Resample if necessary (assuming 16kHz for VAD)
            resampled_audio = resample_audio(float_array, original_sr=48000, target_sr=16000)

            # VAD processing (simplified)
            is_speech = vad.is_speech(resampled_audio.tobytes(), sample_rate=16000)
            if is_speech:
                speech_detected = True
                audio_buffer.extend(float_array)
            elif speech_detected:
                # End of speech segment, process buffer
                transcription = stt_model.transcribe(audio_buffer)
                await websocket.send_json({"type": "transcription", "text": transcription})

                translation = mt_model.translate(transcription, source_lang="en", target_lang="sk")
                await websocket.send_json({"type": "translation", "text": translation})

                # Synthesize audio (example with Piper)
                synthesized_audio_bytes = piper_tts.synthesize(translation, speaker_id="en_US-ryan-medium")
                encoded_audio = base64.b64encode(synthesized_audio_bytes).decode('utf-8')
                await websocket.send_json({"type": "audio", "audio": encoded_audio})

                audio_buffer = []
                speech_detected = False
            else:
                # No speech, clear buffer if too long
                if len(audio_buffer) > 48000 * 5: # e.g., 5 seconds of silence
                    audio_buffer = []

    except WebSocketDisconnect:
        print("WebSocket disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("Cleaning up WebSocket resources.")

```

## 5. Experimental Setup and Results

This chapter details the experimental methodology used to evaluate the performance of the real-time speech translation system, presents the collected results, and discusses their implications regarding the project's objectives.

### 5.1 Experimental Setup

The system was evaluated on a specific hardware and software configuration to ensure consistent and reproducible results.

* **Hardware Environment:**
  * **Development Machine:** Apple MacBook Pro with M1 Pro chip, 16GB Unified Memory, SSD.
  * **Testing Environments:**
    * **macOS:** Host machine (Ventura/Sonoma) for direct testing with BlackHole virtual audio device.
    * **Windows:** Virtual Machine (VM) running Windows (e.g., via UTM) for testing with VB-Cable virtual audio device.
* **Software Environment:**
  * **Operating System:** macOS (Ventura/Sonoma), Windows (within VM).
  * **Python Version:** 3.9+
  * **Key Libraries:** `faster-whisper`, `CTranslate2`, `Piper TTS`, `Coqui TTS`, `FastAPI`, `webrtcvad`, `soundfile`, `torch`, `torchaudio`.
  * **Virtual Environment:** `venv` for dependency management.
  * **Virtual Audio Devices:** BlackHole (macOS), VB-Cable (Windows).
* **Model Configurations:**
  * **STT:** `faster-whisper` (model sizes: `tiny`, `base`, `medium`, `large` for comparison; `base` or `medium` for primary pipeline evaluation), `compute_type="int8"`.
  * **MT:** `CTranslate2` with `Helsinki-NLP Opus-MT` (e.g., `en-sk`, `sk-en`, `en-cs`, `cs-en`).
  * **TTS:** `Piper TTS` (e.g., `en_US-ryan-medium`, `sk_SK-lili-medium`, `cs_CZ-jirka-medium`), `Coqui TTS` (with pre-recorded speaker voice profile).
  * **VAD:** `webrtcvad` (aggressiveness level 3), with pre-VAD RMS silence thresholding.
* **Testing Methodology:**
  * **Functional Testing:**
    * **UI Interactions:** Comprehensive testing of all frontend elements (buttons, dropdowns, modals, theme toggle, language selectors, voice management controls).
    * **Backend API:** Unit and integration tests for user authentication, voice upload, rename, and delete endpoints.
    * **End-to-End Pipeline:** Verification of the entire audio input -> STT -> MT -> TTS -> audio output flow.
  * **Virtual Audio Device Integration Testing:**
    * **Objective:** Confirm that the translated audio output from the web application can be successfully routed to and picked up by virtual audio devices (BlackHole on macOS, VB-Cable on Windows) and subsequently used as a microphone input in third-party conferencing applications (e.g., Google Meet, Zoom).
    * **Procedure:** Manual verification steps involving configuring browser audio output to the virtual device, and then configuring the conferencing application's microphone input to the virtual device's output.
  * **Performance & Latency Testing:**
    * **Latency Measurement:** End-to-end latency (from speech input start to synthesized audio playback start) and component-wise latencies (STT inference, MT inference, TTS inference, network transmission) will be recorded using internal timestamps and external measurements.
    * **Scalability Testing (Simultaneous Translations):** A dedicated test will be created to simulate multiple concurrent users performing real-time translations. This will involve sending multiple audio streams simultaneously to the backend to assess the session-based model management's performance, resource utilization (CPU, memory), and identify potential bottlenecks under load.
    * **TTS Latency:** Measure text-to-audio latency for both Piper and Coqui TTS.
  * **Accuracy Evaluation:**
    * **STT:** Word Error Rate (WER) will be used to evaluate transcription accuracy against ground truth transcripts for various STT model sizes (`tiny`, `base`, `medium`, `large`).
    * **MT:** BLEU (Bilingual Evaluation Understudy) and METEOR (Metric for Evaluation of Translation with Explicit Ordering) scores will be used to assess translation quality against human-translated references.
  * **Audio Quality:** Subjective listening tests will be conducted for TTS output (Piper and Coqui TTS). Objective metrics like Mean Opinion Score (MOS) could be considered for future work.
  * **Test Data:** A diverse set of pre-recorded audio files (e.g., `test/Can you hear me_.wav`, `test/My test speech_xtts_speaker_clean.wav`, `test/slovak_test_speech.wav`) with corresponding ground truth transcripts and translations will be used. Speaker voice profiles will be prepared for Coqui TTS evaluation.

### 5.2 Performance Metrics and Results

The evaluation focused on key performance indicators (KPIs) to assess the system's real-time capabilities and translation quality.

#### 5.2.1 Latency Analysis

The primary objective was to achieve low end-to-end latency. The system's UI includes a real-time timeline chart to visualize latency breakdown, providing immediate feedback on pipeline performance.

* **End-to-End Latency:**

  * **Target:** 2-3 seconds (Piper TTS), 2.5-3.5 seconds (Coqui TTS voice cloning)
  * **Achieved:** [Insert actual measured average latency for Piper TTS] seconds, [Insert actual measured average latency for Coqui TTS] seconds.
  * **Discussion:** Analyze if targets were met. Discuss factors influencing latency (e.g., chunk size, model size, hardware load, network conditions).
* **Component-wise Latency Breakdown:**

  * **STT Inference:** [Insert average STT latency] ms
  * **MT Inference:** [Insert average MT latency] ms
  * **TTS Inference (Piper):** [Insert average Piper TTS latency] ms
  * **TTS Inference (Coqui TTS):** [Insert average Coqui TTS latency] ms
  * **VAD Processing:** [Insert average VAD latency] ms
  * **Network Transmission:** [Insert average network latency] ms
  * **Chart:** A bar chart illustrating the average latency contribution of each pipeline component.

```mermaid
pie
    "STT Inference" : [Insert STT Latency Percentage]
    "MT Inference" : [Insert MT Latency Percentage]
    "TTS Inference (Piper)" : [Insert Piper TTS Latency Percentage]
    "TTS Inference (Coqui TTS)" : [Insert Coqui TTS Latency Percentage]
    "VAD Processing" : [Insert VAD Latency Percentage]
    "Network Transmission" : [Insert Network Latency Percentage]
```

*(Note: The above Mermaid chart is a placeholder. Actual data would be used to generate a visual representation.)*

#### 5.2.2 Accuracy Evaluation

* **STT Accuracy (WER):**
  * **Results:** [Insert average WER for English STT], [Insert average WER for Slovak STT].
  * **Discussion:** Interpret WER scores. Discuss challenges (e.g., accents, background noise) and how `faster-whisper` performed.
* **MT Quality (BLEU/METEOR):**
  * **Results:**
    * English to Slovak: BLEU = [Insert BLEU score], METEOR = [Insert METEOR score]
    * Slovak to English: BLEU = [Insert BLEU score], METEOR = [Insert METEOR score]
  * **Discussion:** Explain BLEU and METEOR scores. Evaluate translation quality and fluency.

#### 5.2.3 TTS Quality Comparison

* **Piper TTS:** [Describe perceived quality: naturalness, expressiveness, clarity]
* **Coqui TTS (Voice Cloning):** [Describe perceived quality: similarity to source voice, naturalness, expressiveness]
* **Discussion:** Compare the trade-offs between Piper's speed and Coqui TTS's voice cloning capabilities.

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

| Metric                                | Value                   |
| ------------------------------------- | ----------------------- |
| **Real-Time Factor (RTF)**      | 1.72                    |
| **First-Chunk Latency**         | 1.21s (optimal config)  |
| **Perceived Latency**           | 2-3s (with streaming)   |
| **Speaker Embedding Cache Hit** | ~1.9% speedup           |
| **Artifact Rate**               | 0% (after optimization) |

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

##### Why Coqui TTS over Coqui TTS?

During development, **Coqui TTS** was initially selected for voice cloning based on its promising zero-shot capabilities. However, after extensive testing, **Coqui TTS (XTTS v2)** was chosen as the final solution for the following reasons:

1. **Stability Issues:**

   - Coqui TTS exhibited frequent crashes on Apple Silicon when using the MPS (Metal Performance Shaders) backend.
   - Fallback to CPU was required, but even then, output quality was inconsistent.
   - Coqui TTS demonstrated robust performance across different hardware configurations.
2. **Official Multilingual Support:**

   - XTTS v2 officially supports 17 languages, including **Czech and Slovak**, which are critical for this project.
   - Coqui TTS's multilingual capabilities were experimental and less mature, requiring extensive tuning for non-English languages.
   - Official support meant better documentation, pre-trained models, and community resources.
3. **Better Documentation and Community:**

   - Coqui TTS benefits from comprehensive documentation, active GitHub community, and regular updates.
   - Coqui TTS, while innovative, had limited documentation and fewer examples for production use.
   - This made debugging and optimization significantly easier with Coqui TTS.
4. **Cross-Lingual Voice Cloning:**

   - XTTS v2 is specifically designed for cross-lingual scenarios—a core requirement for this translation system.
   - It can take a voice sample in English and synthesize speech in Czech with the same voice characteristics.
   - Coqui TTS's cross-lingual capabilities required experimental tuning and produced less consistent results.

**Trade-off:** Coqui TTS has higher latency (RTF 1.72 on CPU) compared to Coqui TTS's theoretical performance, but the stability and quality improvements justified this trade-off.

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

### 5.3 Discussion of Results

Summarize the key findings from the experimental evaluation. Discuss how the system's performance aligns with the initial goals and objectives. Highlight successful aspects, such as the low latency achieved on Apple Silicon and the effective integration of various open-source models. Address any discrepancies or limitations observed during testing and propose potential reasons. For instance, discuss the impact of network conditions on end-to-end latency or the challenges in maintaining high accuracy across diverse linguistic inputs.

## 6. Conclusion and Future Work

### 6.1 Summary of Achievements

This bachelor's thesis successfully designed, implemented, and evaluated a real-time live speech translation system, demonstrating a robust solution for seamless cross-lingual communication. The project achieved its primary objectives by integrating state-of-the-art open-source models for Speech-to-Text (STT), Machine Translation (MT), and Text-to-Speech (TTS) into a low-latency pipeline. Key accomplishments include:

* **Real-time Performance:** The system demonstrated the capability to achieve target end-to-end latencies, making it suitable for live interactive scenarios.
* **Modular and Optimized Architecture:** A flexible client-server architecture with a FastAPI backend and a responsive web frontend was developed, optimized for efficient execution on Apple Silicon hardware.
* **Advanced Model Integration:** Successful integration and optimization of `faster-whisper` for STT, `CTranslate2` with `Helsinki-NLP Opus-MT` for MT, and both `Piper TTS` and `Coqui TTS` for high-quality and voice-cloned speech synthesis.
* **Enhanced User Experience:** The web UI provides dynamic language switching, effective Voice Activity Detection (VAD), and real-time latency visualization, contributing to an intuitive user experience.

The project successfully validated the feasibility of building a high-performance, open-source-driven real-time speech translation system, addressing critical challenges in latency and quality.

### 6.2 Limitations of the Current System

Despite its achievements, the current system has certain limitations that present opportunities for future development:

* **Single-Speaker Focus:** The current voice cloning (Coqui TTS) is primarily designed for a single, pre-defined speaker voice. Multi-speaker scenarios are not yet fully supported, which limits its applicability in complex conference environments.
* **Resource Usage:** While optimized for Apple Silicon, the system can still be resource-intensive, especially when running larger models or multiple instances, which might impact performance on less powerful hardware.
* **Error Handling and Feedback:** The frontend's error handling and user feedback mechanisms could be further enhanced to provide more detailed information during backend initialization failures or network issues.
* **Objective Audio Quality Metrics:** While subjective listening tests were conducted, the integration of objective audio quality metrics (e.g., MOS scores) for TTS output was not fully implemented.

### 6.3 Future Enhancements

Building upon the current foundation, several key areas for future work have been identified to further enhance the system's capabilities and robustness:

* **Multi-speaker Support:** Extend the system to accurately identify and translate speech from multiple speakers simultaneously, potentially incorporating speaker diarization techniques.
* **Production Optimization:** Explore advanced optimization techniques such as model quantization, pruning, and leveraging highly optimized inference engines like `whisper.cpp` or `mlx-whisper` for STT. Investigate cloud deployment options for scalable, production-ready environments.
* **`pip` Packaging:** Simplify the installation and deployment process by packaging the project as a Python library, making it easier for other developers to integrate and use.
* **Advanced Voice Training:** Implement more sophisticated voice training techniques, potentially allowing users to fine-tune TTS models with their own extensive datasets for highly personalized voice cloning.
* **User Management and Profiles:** Develop a comprehensive user authentication and profile management system to store individual language preferences, voice models, and usage history.
* **Improved UI/UX:** Further enhance the frontend with more intuitive controls, richer visual feedback mechanisms, and a more polished, accessible design. This includes refining the latency visualization and adding more interactive elements.
* **Robust Error Handling:** Implement more granular error handling and logging across the entire pipeline, providing clearer diagnostics and recovery mechanisms.
* **Scalability:** Design and implement strategies for horizontal scaling of the backend services to handle a larger number of concurrent users or higher processing loads.
* **Additional Models:** Integrate and evaluate alternative STT, MT, or TTS models to compare performance, explore different linguistic capabilities, or support specialized use cases.

## 7. References

* [1] OpenAI. (2022). *Whisper*. Retrieved from [https://openai.com/research/whisper](https://openai.com/research/whisper)
* [2] CTranslate2. (n.d.). *CTranslate2: Fast inference engine for OpenNMT models*. Retrieved from [https://opennmt.net/CTranslate2/](https://opennmt.net/CTranslate2/)
* [3] Helsinki-NLP. (n.d.). *Opus-MT*. Retrieved from [https://huggingface.co/Helsinki-NLP](https://huggingface.co/Helsinki-NLP)
* [4] Piper TTS. (n.d.). *Piper: A fast, local neural text to speech system*. Retrieved from [https://github.com/rhasspy/piper](https://github.com/rhasspy/piper)
* [5] Coqui TTS. (n.d.). *Coqui TTS: Fast and Flexible Few-shot Text-to-Speech*. Retrieved from [https://github.com/f5-tts/f5-tts](https://github.com/f5-tts/f5-tts) (Note: This is a placeholder, actual Coqui TTS repository might differ or be part of a larger project like Coqui TTS)
* [6] WebRTC. (n.d.). *Voice Activity Detection (VAD)*. Retrieved from [https://webrtc.github.io/webrtc-org/testing/audio-processing/vad/](https://webrtc.github.io/webrtc-org/testing/audio-processing/vad/)
* [7] FastAPI. (n.d.). *FastAPI: A modern, fast (high-performance) web framework for building APIs with Python 3.7+ based on standard Python type hints*. Retrieved from [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
* [8] Chart.js. (n.d.). *Chart.js: Simple, clean and engaging charts for designers and developers*. Retrieved from [https://www.chartjs.org/](https://www.chartjs.org/)
* [9] Apple Inc. (n.d.). *Apple M1, M2, M3 Chips*. Retrieved from [https://www.apple.com/mac/](https://www.apple.com/mac/) (General reference for Apple Silicon)

<style>#mermaid-1763738590133{font-family:sans-serif;font-size:16px;fill:#333;}#mermaid-1763738590133 .error-icon{fill:#552222;}#mermaid-1763738590133 .error-text{fill:#552222;stroke:#552222;}#mermaid-1763738590133 .edge-thickness-normal{stroke-width:2px;}#mermaid-1763738590133 .edge-thickness-thick{stroke-width:3.5px;}#mermaid-1763738590133 .edge-pattern-solid{stroke-dasharray:0;}#mermaid-1763738590133 .edge-pattern-dashed{stroke-dasharray:3;}#mermaid-1763738590133 .edge-pattern-dotted{stroke-dasharray:2;}#mermaid-1763738590133 .marker{fill:#333333;}#mermaid-1763738590133 .marker.cross{stroke:#333333;}#mermaid-1763738590133 svg{font-family:sans-serif;font-size:16px;}#mermaid-1763738590133 .label{font-family:sans-serif;color:#333;}#mermaid-1763738590133 .label text{fill:#333;}#mermaid-1763738590133 .node rect,#mermaid-1763738590133 .node circle,#mermaid-1763738590133 .node ellipse,#mermaid-1763738590133 .node polygon,#mermaid-1763738590133 .node path{fill:#ECECFF;stroke:#9370DB;stroke-width:1px;}#mermaid-1763738590133 .node .label{text-align:center;}#mermaid-1763738590133 .node.clickable{cursor:pointer;}#mermaid-1763738590133 .arrowheadPath{fill:#333333;}#mermaid-1763738590133 .edgePath .path{stroke:#333333;stroke-width:1.5px;}#mermaid-1763738590133 .flowchart-link{stroke:#333333;fill:none;}#mermaid-1763738590133 .edgeLabel{background-color:#e8e8e8;text-align:center;}#mermaid-1763738590133 .edgeLabel rect{opacity:0.5;background-color:#e8e8e8;fill:#e8e8e8;}#mermaid-1763738590133 .cluster rect{fill:#ffffde;stroke:#aaaa33;stroke-width:1px;}#mermaid-1763738590133 .cluster text{fill:#333;}#mermaid-1763738590133 div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:sans-serif;font-size:12px;background:hsl(80,100%,96.2745098039%);border:1px solid #aaaa33;border-radius:2px;pointer-events:none;z-index:100;}#mermaid-1763738590133:root{--mermaid-font-family:sans-serif;}#mermaid-1763738590133:root{--mermaid-alt-font-family:sans-serif;}#mermaid-1763738590133 flowchart{fill:apa;}</style>
