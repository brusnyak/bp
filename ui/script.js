document.addEventListener("DOMContentLoaded", () => {
  // --- DOM Element References ---
  const initBtn = document.getElementById("initBtn");
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");
  const statusIndicator = document.getElementById("statusIndicator");
  const statusLabel = document.getElementById("statusLabel");
  const inputLevelSegmentsContainer =
    document.getElementById("inputLevelSegments");
  const NUM_LEVEL_SEGMENTS = 10;
  let levelSegments = [];
  const ttsModelSelect = document.getElementById("ttsModelSelect");
  const inputLanguageSelect = document.getElementById("inputLanguageSelect");
  const outputLanguageSelect = document.getElementById("outputLanguageSelect");
  const inputLanguageBadge = document.getElementById("inputLanguageBadge");
  const outputLanguageBadge = document.getElementById("outputLanguageBadge");
  const transcriptionBox = document.getElementById("transcriptionBox");
  const translationBox = document.getElementById("translationBox");
  const stateListening = document.getElementById("stateListening");
  const stateTranslating = document.getElementById("stateTranslating");
  const stateSpeaking = document.getElementById("stateSpeaking");
  const settingsStatusText = document.getElementById("settingsStatusText"); // Existing status text
  const activityLog = document.getElementById("activityLog"); // New activity log container

  // --- Latency Metrics Elements ---
  const inputToSttTime = document.getElementById("inputToSttTime");
  const sttToMtTime = document.getElementById("sttToMtTime");
  const mtToTtsTime = document.getElementById("mtToTtsTime");
  const totalPipelineTime = document.getElementById("totalPipelineTime");
  const totalE2ELatency = document.getElementById("totalE2ELatency");

  // --- Latency Tracking State ---
  let lastInputTimestamp = 0;
  let sttReceivedTimestamp = 0;
  let mtReceivedTimestamp = 0;
  let ttsReceivedTimestamp = 0;
  let playbackStartedTimestamp = 0;
  let playbackQueue = []; // Queue for audio buffers
  let isPlayingAudio = false; // Flag to manage sequential playback
  let recordingStartTime = 0; // Absolute timestamp when recording starts for chart X-axis
  let currentOutputChartIndex = -1; // Temporary storage for the chart index of the current audio segment

  // Chart.js instance
  let latencyChart = null; // Chart.js instance
  let chartData = { // Data structure for the chart
    labels: ["Input Speech", "Translated Speech"],
    datasets: [],
  };

  // --- Global State ---
  let websocket = null;
  let audioContext = null;
  let mediaStream = null;
  let audioProcessor = null;
  let isInitialized = false;
  let isRecording = false;
  let blackHoleOutputDeviceId = null; // Store BlackHole device ID for output routing

  // Build API / WS URLs using current page protocol and host (safer on dev vs prod)
  const proto = window.location.protocol === "https:" ? "https" : "http";
  const wsProto = window.location.protocol === "https:" ? "wss" : "ws";
  const API_BASE_URL = `${proto}://${window.location.hostname}${window.location.port ? ":" + window.location.port : ""}`;
  const WS_URL = `${wsProto}://${window.location.hostname}${window.location.port ? ":" + window.location.port : ""}/ws`;
  const API_URL = `${API_BASE_URL}`;

  // Initialize level segments
  for (let i = 0; i < NUM_LEVEL_SEGMENTS; i++) {
    const segment = document.createElement("div");
    segment.classList.add("level-segment");
    inputLevelSegmentsContainer.appendChild(segment);
    levelSegments.push(segment);
  }

  const SAMPLE_RATE = 16000;

  // --- Input level smoothing state ----
  let emaLevel = 0.0; // exponential moving average for smoothing
  const EMA_ALPHA = 0.3; // Increased for more responsiveness

  // --- Throttle state for debug logs (avoid spamming console) ---
  let lastLogAt = 0;
  const LOG_THROTTLE_MS = 1000; // log at most once per second

  // --- Utility Functions ---
  function updateStatus(indicatorClass, labelText) {
    statusIndicator.className = `indicator ${indicatorClass}`;
    statusLabel.textContent = labelText;
  }

  function updateStateIcon(element, isActive) {
    if (isActive) {
      element.classList.remove("idle");
      element.classList.add("active");
    } else {
      element.classList.remove("active");
      element.classList.add("idle");
    }
  }

  function setInputLevel(level) {
    // Expect 'level' to be a 0..1 RMS-like magnitude (server-side should send RMS or average absolute)
    // Apply sqrt to approximate perceptual loudness, then EMA smoothing
    const scaled = Math.sqrt(Math.min(1, Math.max(0, level)));
    emaLevel = emaLevel * (1 - EMA_ALPHA) + scaled * EMA_ALPHA;

    const normalizedLevel = Math.min(1, Math.max(0, emaLevel));
    const activeSegments = Math.ceil(normalizedLevel * NUM_LEVEL_SEGMENTS);

    levelSegments.forEach((segment, index) => {
      if (index < activeSegments) {
        segment.classList.add("active");

        // style classes for medium/high ranges
        if (normalizedLevel > 0.7) {
          segment.classList.add("high");
          segment.classList.remove("medium");
        } else if (normalizedLevel > 0.3) {
          segment.classList.add("medium");
          segment.classList.remove("high");
        } else {
          segment.classList.remove("medium", "high");
        }
      } else {
        segment.classList.remove("active", "medium", "high");
      }
    });
  }

  function resetUI() {
    if (transcriptionBox)
      transcriptionBox.innerHTML = '<p class="placeholder">Waiting for speech...</p>';
    if (translationBox)
      translationBox.innerHTML = '<p class="placeholder">Translation will appear here...</p>';
    emaLevel = 0;
    setInputLevel(0);
    updateStateIcon(stateListening, false);
    updateStateIcon(stateTranslating, false);
    updateStateIcon(stateSpeaking, false);
    if (settingsStatusText) settingsStatusText.textContent = ""; // Clear existing status text
    if (activityLog) activityLog.innerHTML = ""; // Clear activity log

    // Reset latency metrics
    if (inputToSttTime) inputToSttTime.textContent = "0.0s";
    if (sttToMtTime) sttToMtTime.textContent = "0.0s";
    if (ttsToPlaybackTime) ttsToPlaybackTime.textContent = "0.0s";
    if (totalPipelineTime) totalPipelineTime.textContent = "0.0s";
    if (totalE2ELatency) totalE2ELatency.textContent = "0.0s";

    // Reset latency tracking state
    lastInputTimestamp = 0;
    sttReceivedTimestamp = 0;
    mtReceivedTimestamp = 0;
    ttsReceivedTimestamp = 0;
    playbackStartedTimestamp = 0;

    // Clear playback queue
    playbackQueue = [];
    isPlayingAudio = false;

    // Reset chart data
    if (latencyChart) {
      chartData.datasets = [
        {
          label: "Input Speech",
          data: [],
          backgroundColor: "rgba(233, 69, 96, 0.6)", // Red
          borderColor: "rgba(233, 69, 96, 1)",
          borderWidth: 2, // 2px line
          type: 'scatter', // Use scatter for custom drawing
          pointRadius: 0, // No points
          yAxisID: 'y',
        },
        {
          label: "Translated Speech",
          data: [],
          backgroundColor: "rgba(15, 52, 96, 0.6)", // Blue
          borderColor: "rgba(15, 52, 96, 1)",
          borderWidth: 2, // 2px line
          type: 'scatter', // Use scatter for custom drawing
          pointRadius: 0, // No points
          yAxisID: 'y',
        },
      ];
      latencyChart.options.scales.x.max = 300; // Reset max X-axis to 5 minutes
      latencyChart.options.scales.x.min = 0; // Ensure min X-axis is reset to 0
      latencyChart.update();
    }
  }

  function appendLog(message, type = "info") {
    const timestamp = new Date().toLocaleTimeString();
    const logEntry = document.createElement("p");
    logEntry.classList.add("log-entry", `log-${type}`);
    logEntry.innerHTML = `[${escapeHtml(message)}]`;
    activityLog.prepend(logEntry); // Add to top
    // Optional: Limit log entries to prevent UI clutter
    while (activityLog.children.length > 50) {
      activityLog.removeChild(activityLog.lastChild);
    }
  }

  // --- WebSocket Handling ---
  function connectWebSocket() {
    try {
      websocket = new WebSocket(WS_URL);
    } catch (err) {
      settingsStatusText.textContent = `Invalid WS URL: ${WS_URL}`;
      updateStatus("error", "WS Error");
      return;
    }

    websocket.onopen = () => {
      console.log("WebSocket connected.");
      updateStatus("on", "Connected");
      appendLog("WebSocket connected.", "success");
    };

    websocket.onmessage = async (event) => {
      if (event.data instanceof ArrayBuffer || event.data instanceof Blob) {
        // If it's a Blob, convert it to ArrayBuffer first
        const audioData =
          event.data instanceof Blob
            ? await event.data.arrayBuffer()
            : event.data;
        // Pass the stored outputChartIndex with the audio data
        await playAudio(audioData, currentOutputChartIndex);
        currentOutputChartIndex = -1; // Reset after use
      } else {
        try {
          const data = JSON.parse(event.data);
          const currentTime = performance.now(); // Use high-resolution time

          if (data.type === "audio_level") {
            setInputLevel(typeof data.level === "number" ? data.level : 0);
            // Update lastInputTimestamp only if we are actively listening and it's the start of a new segment
            if (isRecording && lastInputTimestamp === 0) {
              lastInputTimestamp = currentTime;
            }
          } else if (data.type === "transcription_result") {
            if (data.transcribed) {
              if (transcriptionBox)
                transcriptionBox.innerHTML = `<p>${escapeHtml(data.transcribed)}</p>`;
              updateStateIcon(stateTranslating, true);
              sttReceivedTimestamp = currentTime;
              if (lastInputTimestamp && inputToSttTime) {
                inputToSttTime.textContent = `${(
                  (sttReceivedTimestamp - lastInputTimestamp) /
                  1000
                ).toFixed(2)}s`;
              }
            }
          } else if (data.type === "translation_result") {
            if (data.translated) {
              if (translationBox)
                translationBox.innerHTML = `<p>${escapeHtml(data.translated)}</p>`;
              updateStateIcon(stateTranslating, false);
              updateStateIcon(stateSpeaking, true);
              mtReceivedTimestamp = currentTime;
              if (sttReceivedTimestamp && sttToMtTime) {
                sttToMtTime.textContent = `${(
                  (mtReceivedTimestamp - sttReceivedTimestamp) /
                  1000
                ).toFixed(2)}s`;
              }
            }
          } else if (data.type === "final_metrics") {
            if (data.metrics) {
              updateStateIcon(stateSpeaking, false);
              ttsReceivedTimestamp = currentTime; // This is when TTS audio is *sent* from backend
              if (mtReceivedTimestamp && mtToTtsTime) {
                mtToTtsTime.textContent = `${(
                  (ttsReceivedTimestamp - mtReceivedTimestamp) /
                  1000
                ).toFixed(2)}s`;
              }
              // Total pipeline time is STT+MT+TTS from backend
              if (totalPipelineTime)
                totalPipelineTime.textContent = `${(
                  data.metrics.stt_time + data.metrics.mt_time + data.metrics.tts_time
                ).toFixed(2)}s`;

              // Add data to chart
              if (lastInputTimestamp && recordingStartTime && latencyChart) {
                const inputStartRelative =
                  (lastInputTimestamp - recordingStartTime) / 1000;
                const inputEndRelative =
                  (sttReceivedTimestamp - recordingStartTime) / 1000;
                const outputStartRelative =
                  (ttsReceivedTimestamp - recordingStartTime) / 1000;

                console.log(`[Chart Debug] Adding input data:
                  inputStartRelative: ${inputStartRelative.toFixed(2)}s
                  inputEndRelative: ${inputEndRelative.toFixed(2)}s`);
                console.log(`[Chart Debug] Adding output placeholder:
                  outputStartRelative: ${outputStartRelative.toFixed(2)}s`);

                // Ensure datasets exist
                if (!chartData.datasets[0]) {
                  chartData.datasets[0] = {
                    label: "Input Speech",
                    data: [],
                    backgroundColor: "rgba(233, 69, 96, 0.6)", // Red
                    borderColor: "rgba(233, 69, 96, 1)",
                    borderWidth: 8, // 8px line for better visibility
                    type: 'scatter', // Use scatter for custom drawing
                    pointRadius: 0, // No points
                    yAxisID: 'y',
                  };
                }
                if (!chartData.datasets[1]) {
                  chartData.datasets[1] = {
                    label: "Translated Speech",
                    data: [],
                    backgroundColor: "rgba(250, 202, 6, 0.6)", // Yellow for better visibility
                    borderColor: "rgba(252, 232, 4, 1)",
                    borderWidth: 8, // 8px line for better visibility
                    type: 'scatter', // Use scatter for custom drawing
                    pointRadius: 0, // No points
                    yAxisID: 'y',
                  };
                }

                // Add data to chart for input speech
                const inputChartIndex = chartData.datasets[0].data.length;
                const newInputData = {
                  x: inputStartRelative,
                  y: "Input Speech",
                  x2: inputEndRelative,
                  index: inputChartIndex // Ensure index is added here
                };
                chartData.datasets[0].data.push(newInputData);
                console.log(`[Chart Debug] Pushed input data: ${JSON.stringify(newInputData)}`);

                // Add a placeholder for translated speech, will update x2 when audio finishes playing
                const outputChartIndex = chartData.datasets[1].data.length;
                chartData.datasets[1].data.push({
                  x: outputStartRelative,
                  y: "Translated Speech",
                  x2: outputStartRelative, // Placeholder, will be updated
                  index: outputChartIndex // Store index for later update
                });

                // Store the outputChartIndex globally for the next incoming audio data
                currentOutputChartIndex = outputChartIndex;

                // Dynamically update X-axis max
                const currentChartMaxX = latencyChart.options.scales.x.max;
                const latestEndTime = Math.max(inputEndRelative, outputStartRelative); // Use outputStartRelative for now

                // Expand max X-axis if the latest event exceeds the current max, with a small buffer
                if (latestEndTime + 10 > currentChartMaxX) { // Add 10 seconds buffer
                  latencyChart.options.scales.x.max = latestEndTime + 10;
                }

                // Ensure min X-axis remains at 0 for continuous timeline
                latencyChart.options.scales.x.min = 0;
                
                latencyChart.update();
              }

              // Reset lastInputTimestamp for the next speech segment
              lastInputTimestamp = 0;
            }
          } else if (data.type === "status") {
            // Backend status messages go to the new activity log
            appendLog(data.message || "", "info");
          } else if (data.status === "error") {
            console.error("Server Error:", data.message);
            appendLog(`Server Error: ${data.message}`, "error");
            if (settingsStatusText)
              settingsStatusText.textContent = `Error: ${data.message}`; // Keep critical error in settingsStatusText
            stopRecording();
          }
        } catch (error) {
          console.error("Error parsing WebSocket message:", error);
          appendLog(
            `Error parsing WebSocket message: ${error.message}`,
            "error",
          );
        }
      }
    };

    websocket.onclose = () => {
      console.log("WebSocket disconnected.");
      updateStatus("off", "Disconnected");
      appendLog("WebSocket disconnected.", "warning");
      stopRecording();
    };

    websocket.onerror = (error) => {
      console.error("WebSocket Error:", error);
      settingsStatusText.textContent = "WebSocket connection error.";
      updateStatus("error", "Error");
      appendLog(`WebSocket Error: ${error.message}`, "error");
      stopRecording();
    };
  }

  function closeWebSocket() {
    if (websocket) {
      websocket.close();
      websocket = null;
    }
  }

  // Modified playAudio to accept audio data and its corresponding chart index
  async function playAudio(audioBuffer, outputChartIndex) {
    playbackQueue.push({ audioBuffer, outputChartIndex });
    if (!isPlayingAudio) {
      playNextAudioInQueue();
    }
  }

  async function playNextAudioInQueue() {
    if (playbackQueue.length === 0 || isPlayingAudio) {
      return;
    }

    isPlayingAudio = true;
    const { audioBuffer, outputChartIndex } = playbackQueue.shift(); // Get the next audio buffer and its index

    let playbackAudioContext;
    try {
      playbackAudioContext = new (
        window.AudioContext || window.webkitAudioContext
      )({
        sampleRate: SAMPLE_RATE,
        sinkId: blackHoleOutputDeviceId || "default",
      });
      console.log(
        `Playback AudioContext created with sinkId: ${playbackAudioContext.sinkId}. Actual sinkId used: ${playbackAudioContext.sinkId}`,
      );
      appendLog(
        `Playback AudioContext created. Attempted sinkId: ${blackHoleOutputDeviceId || "default"}. Actual sinkId: ${playbackAudioContext.sinkId}`,
        "info",
      );
    } catch (e) {
      console.error("Error creating playback AudioContext:", e);
      appendLog(`Error creating playback AudioContext: ${e.message}`, "error");
      isPlayingAudio = false;
      playNextAudioInQueue();
      return;
    }

    try {
      const audioData = await playbackAudioContext.decodeAudioData(audioBuffer);
      const source = playbackAudioContext.createBufferSource();
      source.buffer = audioData;
      source.connect(playbackAudioContext.destination);
      source.start(0);
      playbackStartedTimestamp = performance.now(); // Mark when playback actually starts

      // Update E2E latency and TTS to Playback time
      if (lastInputTimestamp && totalE2ELatency) {
        totalE2ELatency.textContent = `${(
          (playbackStartedTimestamp - lastInputTimestamp) /
          1000
        ).toFixed(2)}s`;
      }
      if (ttsReceivedTimestamp && ttsToPlaybackTime) {
        ttsToPlaybackTime.textContent = `${(
          (playbackStartedTimestamp - ttsReceivedTimestamp) /
          1000
        ).toFixed(2)}s`;
      }

      appendLog(
        `Playing synthesized audio to ${blackHoleOutputDeviceId ? "BlackHole" : "default output"}.`,
        "success",
      );

      source.onended = () => {
        playbackAudioContext.close();
        isPlayingAudio = false;

        // Update the x2 (end time) for the corresponding translated speech bar
        if (latencyChart && outputChartIndex !== undefined && chartData.datasets[1].data[outputChartIndex]) {
          const outputEndRelative = (performance.now() - recordingStartTime) / 1000;
          chartData.datasets[1].data[outputChartIndex].x2 = outputEndRelative;
          console.log(`[Chart Debug] Updated output bar ${outputChartIndex} x2 to: ${outputEndRelative.toFixed(2)}s`);
          latencyChart.update();
        }
        playNextAudioInQueue();
      };
    } catch (error) {
      console.error("Error playing audio:", error);
      appendLog(`Error playing audio: ${error.message}`, "error");
      isPlayingAudio = false;
      playNextAudioInQueue();
    }
  }

  // --- Audio Recording Handling ---
  async function startRecording() {
    if (isRecording) return;

    // Reset chart data on new recording start
    if (latencyChart) {
      chartData.datasets = []; // Clear all datasets
      latencyChart.options.scales.x.max = 300; // Reset max X-axis to 5 minutes
      latencyChart.options.scales.x.min = 0; // Ensure min X-axis is reset to 0
      latencyChart.update();
    }

    // Reset recordingStartTime and playbackStartedTimestamp for each new recording session
    recordingStartTime = performance.now(); // Set absolute recording start time
    playbackStartedTimestamp = 0; // Reset playback start time
    console.log(`[Chart Debug] Recording started. recordingStartTime: ${recordingStartTime}`);

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      if (settingsStatusText)
        settingsStatusText.textContent =
          "Microphone not supported by your browser.";
      updateStatus("error", "Mic Error");
      appendLog("Microphone not supported by browser.", "error");
      return;
    }

    try {
      // Remove hardcoded BlackHole input. Use default microphone.
      // We still enumerate devices to find BlackHole for output later.
      const devices = await navigator.mediaDevices.enumerateDevices();
      const blackHoleOutputDevice = devices.find(
        (device) =>
          device.kind === "audiooutput" &&
          device.label.toLowerCase().includes("blackhole"),
      );

      if (blackHoleOutputDevice) {
        blackHoleOutputDeviceId = blackHoleOutputDevice.deviceId;
        appendLog(
          `Found BlackHole 2ch output device: ${blackHoleOutputDevice.label}`,
          "info",
        );
      } else {
        blackHoleOutputDeviceId = null;
        appendLog(
          "Warning: BlackHole 2ch output device not found. Translated audio will play through default output.",
          "warning",
        );
      }

      let audioConstraints = {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      };

      // No specific deviceId for input, will use default microphone
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: audioConstraints,
      });

      appendLog("Using default microphone for input.", "info");
      if (settingsStatusText)
        settingsStatusText.textContent = "Microphone set to: Default";

      audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: SAMPLE_RATE,
      });

      // Resume audio context if it was suspended
      if (audioContext.state === "suspended") {
        await audioContext.resume();
      }

      const source = audioContext.createMediaStreamSource(mediaStream);

      try {
        // Load AudioWorklet processor
        await audioContext.audioWorklet.addModule("ui/audio-processor.js");
        audioProcessor = new AudioWorkletNode(audioContext, "audio-processor");

        audioProcessor.port.onmessage = (event) => {
          if (websocket && websocket.readyState === WebSocket.OPEN) {
          const audioData = event.data;
          const audioClone = new Float32Array(audioData); // copy before sending

          // Throttled debug log to avoid heavy console spam
          const now = Date.now();
          if (now - lastLogAt > LOG_THROTTLE_MS) {
            console.debug(`Received ${audioClone.length} audio samples from AudioWorklet. Sending to WebSocket.`);
            lastLogAt = now;
          }

          // Send as binary message (server must accept Float32Array buffer)
          websocket.send(audioClone.buffer);
          }
        };
      } catch (workletError) {
        console.error("Error loading AudioWorklet:", workletError);
        settingsStatusText.textContent = `Audio processing error: ${workletError.message}`;
        appendLog(`AudioWorklet failed to load: ${workletError.message}`, "error");
        stopRecording();
        return;
      }

      source.connect(audioProcessor);
      // audioProcessor -> destination is required for some browsers to keep the processing alive
      // audioProcessor.connect(audioContext.destination); // Removed to prevent echo

      isRecording = true;
      updateStatus("on", "Listening...");
      startBtn.disabled = true;
      stopBtn.disabled = false;
      updateStateIcon(stateListening, true);

      if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({ type: "start" }));
      }
    } catch (error) {
      console.error("Error accessing microphone:", error);
      if (settingsStatusText)
        settingsStatusText.textContent = `Microphone access denied: ${error.message}`;
      updateStatus("error", "Mic Error");
      stopRecording();
    }
  }

  function stopRecording() {
    if (!isRecording) return;

    isRecording = false;
    updateStatus("off", "Ready");
    startBtn.disabled = false;
    stopBtn.disabled = true;
    updateStateIcon(stateListening, false);
    updateStateIcon(stateTranslating, false);
    updateStateIcon(stateSpeaking, false);
    resetUI();

    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStream = null;
    }
    if (audioProcessor) {
      try {
        audioProcessor.port.onmessage = null; // Clear message handler
        audioProcessor.disconnect();
      } catch (_) {
        /* ignore */
      }
      audioProcessor = null;
    }
    // Ensure audioContext is closed to stop all processing and release resources
    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }

    // Send stop command to backend AFTER clearing local audio processing
    if (websocket && websocket.readyState === WebSocket.OPEN) {
      websocket.send(JSON.stringify({ type: "stop" }));
    }
  }

  // --- Helpers & Events ---
  initBtn.addEventListener("click", async () => {
    if (isInitialized) {
      if (settingsStatusText)
        settingsStatusText.textContent = "Pipeline already initialized.";
      return;
    }
    initBtn.disabled = true;
    updateStatus("prepping", "Initializing Models...");
    appendLog("Initializing models...", "info");
    if (settingsStatusText)
      settingsStatusText.textContent =
        "Loading models (this may take ~1 minute on first run)..."; // Keep this for prominent display

    try {
      const sourceLang = inputLanguageSelect.value;
      const targetLang = outputLanguageSelect.value;
      const ttsModel = ttsModelSelect.value;

      appendLog(
        `Attempting to initialize pipeline with: Source=${sourceLang.toUpperCase()}, Target=${targetLang.toUpperCase()}, TTS Model=${ttsModel}.`,
        "info",
      );

      const response = await fetch(
        `${API_URL}/initialize?source_lang=${sourceLang}&target_lang=${targetLang}&tts_model_choice=${ttsModel}`,
        { method: "POST" },
      );
      const data = await response.json();

      if (data.status === "success") {
        isInitialized = true;
        startBtn.disabled = false;
        updateStatus("off", "Ready");
        if (settingsStatusText) settingsStatusText.textContent = data.message;
        appendLog(
          `Pipeline initialized successfully: ${data.message}`,
          "success",
        );
        connectWebSocket();
      } else {
        if (settingsStatusText)
          settingsStatusText.textContent = `Initialization failed: ${data.message}`;
        updateStatus("error", "Init Error");
        appendLog(`Initialization failed: ${data.message}`, "error");
        initBtn.disabled = false;
      }
    } catch (error) {
      console.error("Initialization API error:", error);
      if (settingsStatusText)
        settingsStatusText.textContent = `Initialization failed: ${error.message}`;
      updateStatus("error", "Init Error");
      appendLog(`Initialization API error: ${error.message}`, "error");
      initBtn.disabled = false;
    }
  });

  startBtn.addEventListener("click", startRecording);
  stopBtn.addEventListener("click", stopRecording);

  function sendConfigUpdate() {
    if (websocket && websocket.readyState === WebSocket.OPEN) {
      const config = {
        type: "config_update",
        source_lang: inputLanguageSelect.value,
        target_lang: outputLanguageSelect.value,
        tts_model_choice: ttsModelSelect.value,
      };
      websocket.send(JSON.stringify(config));
      appendLog(
        `Configuration updated: Source=${config.source_lang.toUpperCase()}, Target=${config.target_lang.toUpperCase()}, TTS Model=${config.tts_model_choice}.`,
        "info",
      );
      if (settingsStatusText)
        settingsStatusText.textContent = "Configuration updated."; // Keep this for immediate feedback
    } else {
      if (settingsStatusText)
        settingsStatusText.textContent =
          "Not connected to server. Cannot update config.";
      appendLog("Not connected to server. Cannot update config.", "warning");
    }
  }

  inputLanguageSelect.addEventListener("change", (event) => {
    inputLanguageBadge.textContent = event.target.value.toUpperCase();
    appendLog(
      `Input language changed to ${event.target.value.toUpperCase()}.`,
      "info",
    );
    sendConfigUpdate();
  });

  outputLanguageSelect.addEventListener("change", (event) => {
    outputLanguageBadge.textContent = event.target.value.toUpperCase();
    appendLog(
      `Output language changed to ${event.target.value.toUpperCase()}.`,
      "info",
    );
    sendConfigUpdate();
  });

  ttsModelSelect.addEventListener("change", (event) => {
    appendLog(`TTS Model changed to ${event.target.value}.`, "info");
    sendConfigUpdate();
  });

  // Helper function to escape HTML for safe display
  function escapeHtml(unsafe) {
    return unsafe
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Initial UI setup
  resetUI();
  updateStatus("off", "Awaiting Initialization");
  startBtn.disabled = true;
  stopBtn.disabled = true;

  // --- Charting Implementation ---
  // Custom Chart.js plugin for the real-time vertical line
  const realtimeLinePlugin = {
    id: 'realtimeLine',
    beforeDraw: (chart) => {
      if (!isRecording || recordingStartTime === 0) {
        return;
      }

      const { ctx, chartArea: { left, right, top, bottom }, scales: { x } } = chart;
      ctx.save();

      // Draw the line
      ctx.beginPath();
      ctx.strokeStyle = 'rgba(255, 255, 0, 0.8)'; // Yellow line
      ctx.lineWidth = 2;

      const currentTimeRelative = (performance.now() - recordingStartTime) / 1000;
      const xCoordinate = x.getPixelForValue(currentTimeRelative);

      // Ensure the line is within the chart area
      if (xCoordinate >= left && xCoordinate <= right) {
        ctx.moveTo(xCoordinate, top);
        ctx.lineTo(xCoordinate, bottom);
      }
      ctx.stroke();
      ctx.restore();
    },
  };

  // Custom Chart.js plugin for drawing horizontal lines (segments)
  const segmentDrawingPlugin = {
    id: 'segmentDrawing',
    beforeDatasetsDraw: (chart, args, pluginOptions) => {
      const { ctx, chartArea: { left, right, top, bottom }, scales: { x, y } } = chart;
      ctx.save();

      chart.data.datasets.forEach((dataset) => {
        if (dataset.type === 'scatter') { // Only apply to scatter datasets
          const isInput = dataset.label === "Input Speech";
          const yCategory = isInput ? "Input Speech" : "Translated Speech";
          const base_yCenter = y.getPixelForValue(yCategory);
          const lineWidth = dataset.borderWidth || 2; // Use dataset's borderWidth or default to 2px
          const offsetPerSegment = 5; // Adjust this value for desired spacing

          dataset.data.forEach(dataPoint => {
            const yOffset = (dataPoint.index % 2) * offsetPerSegment; // Alternate offset for better visibility
            const yCenter = base_yCenter + yOffset;

            const startX = x.getPixelForValue(dataPoint.x);
            const endX = x.getPixelForValue(dataPoint.x2);

            console.log(`[Segment Drawing Debug] Dataset: ${dataset.label}, Index: ${dataPoint.index},
              yCategory: ${yCategory}, base_yCenter: ${base_yCenter}, yOffset: ${yOffset}, yCenter: ${yCenter},
              startX: ${startX}, endX: ${endX}, dataPoint.x: ${dataPoint.x}, dataPoint.x2: ${dataPoint.x2}`);

            ctx.beginPath();
            ctx.strokeStyle = dataset.borderColor;
            ctx.lineWidth = lineWidth;
            ctx.moveTo(startX, yCenter);
            ctx.lineTo(endX, yCenter);
            ctx.stroke();
          });
        }
      });
      ctx.restore();
    }
  };

  // Set default locale for Chart.js to prevent TypeError
  if (typeof Chart !== 'undefined') {
    Chart.defaults.locale = 'en-US';
  }

  // Initialize Chart.js
  const chartCanvas = document.getElementById("latencyTimelineChart");
  if (chartCanvas) {
    const ctx = chartCanvas.getContext("2d");
    latencyChart = new Chart(ctx, {
      type: "scatter", // Changed to scatter type
      data: chartData,
      options: {
        indexAxis: "y", // Keep y-axis as category for labels
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: "linear",
            position: "bottom",
            title: {
              display: true,
              text: "Time (seconds from start)",
              color: "#e0e0e0",
            },
            grid: {
              color: "rgba(255, 255, 255, 0.1)",
            },
            ticks: {
              color: "#e0e0e0",
            },
            min: 0,
            max: 300, // Initial max set to 5 minutes (300 seconds)
            beginAtZero: true,
          },
          y: {
            type: "category",
            labels: ["Input Speech", "Translated Speech"], // Explicit labels for clarity
            offset: true,
            stacked: false,
            title: {
              display: true,
              text: "Pipeline Stage",
              color: "#e0e0e0",
            },
            grid: {
              color: "rgba(255, 255, 255, 0.1)",
            },
            ticks: {
              color: "#e0e0e0",
            },
          },
        },
        plugins: {
          legend: {
            display: true,
            labels: {
              color: "#e0e0e0",
              filter: function(legendItem, chartData) {
                // Hide legend items for scatter datasets as they are drawn by custom plugin
                return chartData.datasets[legendItem.datasetIndex].type !== 'scatter';
              }
            },
          },
          title: {
            display: true,
            text: "Speech Processing Timeline",
            color: "#e0e0e0",
            font: {
              size: 16,
            },
          },
          tooltip: {
            callbacks: {
              label: function (context) {
                const label = context.dataset.label || "";
                const start = context.raw.x.toFixed(2);
                const end = context.raw.x2 !== undefined ? context.raw.x2.toFixed(2) : start;
                const duration = (end - start).toFixed(2);

                let tooltipText = `${label}: ${start}s - ${end}s (${duration}s)`;

                // If this is an input segment, try to find a corresponding output segment for latency
                if (context.dataset.label === "Input Speech") {
                  const inputSegmentIndex = context.dataIndex;
                  // Find the corresponding output segment (assuming 1:1 mapping and same order)
                  if (chartData.datasets[1].data[inputSegmentIndex]) {
                    const outputStart = chartData.datasets[1].data[inputSegmentIndex].x;
                    const latency = (outputStart - start).toFixed(2);
                    tooltipText += ` | Latency to Output: ${latency}s`;
                  }
                }
                return tooltipText;
              },
            },
          },
          realtimeLine: realtimeLinePlugin, // Register the custom plugin here
          segmentDrawing: segmentDrawingPlugin, // Register the custom segment drawing plugin
        },
      },
      plugins: [realtimeLinePlugin, segmentDrawingPlugin] // Register plugins at the chart level
    });

    // Update the real-time line every 100ms for smoother movement
    setInterval(() => {
      if (isRecording && recordingStartTime > 0 && latencyChart) {
        latencyChart.update();
      }
    }, 100); 

    console.log("Chart.js initialized successfully.");
    console.log("Initial chartData:", chartData);
  } else {
    console.error("Chart canvas element not found!");
    appendLog("Error: Chart canvas element not found.", "error");
  }
});
