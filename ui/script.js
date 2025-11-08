document.addEventListener("DOMContentLoaded", () => {
  // --- DOM Element References ---
  const burgerMenuBtn = document.getElementById("burgerMenuBtn");
  const sidebar = document.getElementById("sidebar");
  const closeSidebarBtn = document.getElementById("closeSidebarBtn");
  const voiceList = document.getElementById("voiceList"); // For displaying stored voices

  const initBtn = document.getElementById("initBtn");
  const recordVoiceBtn = document.getElementById("recordVoiceBtn"); // New button for recording voice
  const startStopBtn = document.getElementById("startStopBtn"); // New Start/Stop button
  const statusIndicator = document.getElementById("statusIndicator");
  const statusLabel = document.getElementById("statusLabel");
  const inputLevelSegmentsContainer =
    document.getElementById("inputLevelSegments");
  const NUM_LEVEL_SEGMENTS = 20;
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

  // Voice Configuration Modal Elements
  const voiceConfigModal = document.getElementById("voiceConfigModal");
  const closeModalBtn = document.getElementById("closeModalBtn");
  const agreeToRecordCheckbox = document.getElementById("agreeToRecordCheckbox");
  const readingText = document.getElementById("readingText");
  const startStopRecordingModalBtn = document.getElementById("startStopRecordingModalBtn"); // Combined Start/Stop button for modal
  const modalRecordingStatus = document.getElementById("modalRecordingStatus");
  const speakerVoiceUploadModal = document.getElementById("speakerVoiceUploadModal");
  const uploadSpeakerVoiceModalBtn = document.getElementById("uploadSpeakerVoiceModalBtn");
  const currentSpeakerVoiceModal = document.getElementById("currentSpeakerVoiceModal");
  const cancelModalBtn = document.getElementById("cancelModalBtn");
  const modalInputLevelSegmentsContainer = document.getElementById("modalInputLevelSegments"); // Mic level for modal
  let modalLevelSegments = []; // Segments for modal mic level
  const modalInputLanguageSelect = document.getElementById("modalInputLanguageSelect"); // New: Language select for modal

  // Voice Naming Modal Elements
  const voiceNamingModal = document.getElementById("voiceNamingModal");
  const closeNamingModalBtn = document.getElementById("closeNamingModalBtn");
  const voiceNameInput = document.getElementById("voiceNameInput");
  const voiceNameError = document.getElementById("voiceNameError");
  const saveVoiceNameBtn = document.getElementById("saveVoiceNameBtn");
  const cancelVoiceNameBtn = document.getElementById("cancelVoiceNameBtn");

  // Delete Confirmation Modal Elements
  const deleteConfirmationModal = document.getElementById("deleteConfirmationModal");
  const closeDeleteModalBtn = document.getElementById("closeDeleteModalBtn");
  const deleteConfirmationText = document.getElementById("deleteConfirmationText");
  const confirmDeleteBtn = document.getElementById("confirmDeleteBtn");
  const cancelDeleteBtn = document.getElementById("cancelDeleteBtn");

  let uploadedSpeakerVoicePath = null; // To store the path of the uploaded speaker WAV
  let isRecordingModal = false; // State for recording within the modal
  let tempUploadedFileName = null; // To store the temporary name of the uploaded file before renaming
  let voiceToDelete = null; // Store the filename of the voice to be deleted

  // Flag to control initial modal display
  let hasShownVoiceConfigModal = false;

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
    
    // Reset modal specific UI elements
    if (currentSpeakerVoiceModal) currentSpeakerVoiceModal.textContent = "No voice selected";
    uploadedSpeakerVoicePath = null;
    if (speakerVoiceUploadModal) speakerVoiceUploadModal.value = ""; // Clear file input
    if (uploadSpeakerVoiceModalBtn) uploadSpeakerVoiceModalBtn.disabled = true;
    if (agreeToRecordCheckbox) agreeToRecordCheckbox.checked = false;
    if (startStopRecordingModalBtn) {
      startStopRecordingModalBtn.disabled = true;
      startStopRecordingModalBtn.textContent = "Start Recording";
      startStopRecordingModalBtn.classList.remove("btn-stop");
      startStopRecordingModalBtn.classList.add("btn-primary");
    }
    if (modalRecordingStatus) modalRecordingStatus.textContent = "Ready to record.";
    if (modalInputLevelSegmentsContainer) {
      modalInputLevelSegmentsContainer.innerHTML = ''; // Clear existing segments
      modalLevelSegments = []; // Reset array
      for (let i = 0; i < NUM_LEVEL_SEGMENTS; i++) {
        const segment = document.createElement("div");
        segment.classList.add("level-segment");
        modalInputLevelSegmentsContainer.appendChild(segment);
        modalLevelSegments.push(segment);
      }
      setInputLevelModal(0); // Reset modal mic level
    }

    // Reset main control buttons
    if (recordVoiceBtn) recordVoiceBtn.disabled = true;
    if (initBtn) initBtn.disabled = false; // Re-enable init button on full UI reset
    if (startStopBtn) {
      startStopBtn.disabled = true;
      startStopBtn.textContent = "Start";
      startStopBtn.classList.remove("btn-stop");
      startStopBtn.classList.add("btn-primary");
    }

    // Reset latency metrics
    if (inputToSttTime) inputToSttTime.textContent = "0.0s";
    if (sttToMtTime) sttToMtTime.textContent = "0.0s";
    if (mtToTtsTime) mtToTtsTime.textContent = "0.0s";
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
          backgroundColor: "rgba(106, 13, 173, 0.6)", // Primary color (Deep Purple)
          borderColor: "rgba(106, 13, 173, 1)",
          borderWidth: 8, // Thicker line for better visibility
          type: 'scatter',
          pointRadius: 0,
          yAxisID: 'y',
        },
        {
          label: "Translated Speech",
          data: [],
          backgroundColor: "rgba(0, 188, 212, 0.6)", // Accent color (Cyan)
          borderColor: "rgba(0, 188, 212, 1)",
          borderWidth: 8, // Thicker line for better visibility
          type: 'scatter',
          pointRadius: 0,
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
      // Enable record voice button after connection and initialization
      if (isInitialized) {
        recordVoiceBtn.disabled = false;
      }
    };

    websocket.onmessage = async (event) => {
      if (event.data instanceof ArrayBuffer || event.data instanceof Blob) {
        const audioData =
          event.data instanceof Blob
            ? await event.data.arrayBuffer()
            : event.data;
        // The backend now sends `is_final` in the `final_metrics` message,
        // so we need to ensure `currentOutputChartIndex` is correctly set
        // before calling `playAudio` if it's a new segment.
        // For now, we'll assume `currentOutputChartIndex` is set by the `final_metrics` handler
        // for the *start* of a new segment, and `playAudio` will update its end.
        await playAudio(audioData, currentOutputChartIndex);
        // Do NOT reset currentOutputChartIndex here. It should be managed by the metrics.
        // currentOutputChartIndex = -1;
      } else {
        try {
          const data = JSON.parse(event.data);
          const currentTime = performance.now();

          if (data.type === "audio_level") {
            const level = typeof data.level === "number" ? data.level : 0;
            setInputLevel(level); // Update main UI mic level
            if (isRecordingModal) {
              setInputLevelModal(level); // Update modal mic level if modal is open
            }
            if (isRecording && lastInputTimestamp === 0) {
              lastInputTimestamp = currentTime;
            }
          } else if (data.type === "transcription_result") {
            if (data.transcribed) {
              if (transcriptionBox) {
                if (data.is_final) {
                  transcriptionBox.innerHTML = `<p>${escapeHtml(data.transcribed)}</p>`;
                } else {
                  transcriptionBox.innerHTML = `<p class="partial-transcription">${escapeHtml(data.transcribed)}</p>`;
                }
              }
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
              if (translationBox) {
                if (data.is_final) {
                  translationBox.innerHTML = `<p>${escapeHtml(data.translated)}</p>`;
                } else {
                  translationBox.innerHTML = `<p class="partial-translation">${escapeHtml(data.translated)}</p>`;
                }
              }
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
              ttsReceivedTimestamp = currentTime;
              if (mtReceivedTimestamp && mtToTtsTime) {
                mtToTtsTime.textContent = `${(
                  (ttsReceivedTimestamp - mtReceivedTimestamp) /
                  1000
                ).toFixed(2)}s`;
              }
              if (totalPipelineTime)
                totalPipelineTime.textContent = `${(
                  data.metrics.stt_time + data.metrics.mt_time + data.metrics.tts_time
                ).toFixed(2)}s`;

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

                if (!chartData.datasets[0]) {
                  chartData.datasets[0] = {
                    label: "Input Speech",
                    data: [],
                    backgroundColor: "rgba(106, 13, 173, 0.6)", // Primary color (Deep Purple)
                    borderColor: "rgba(106, 13, 173, 1)",
                    borderWidth: 8,
                    type: 'scatter',
                    pointRadius: 0,
                    yAxisID: 'y',
                  };
                }
                if (!chartData.datasets[1]) {
                  chartData.datasets[1] = {
                    label: "Translated Speech",
                    data: [],
                    backgroundColor: "rgba(0, 188, 212, 0.6)", // Accent color (Cyan)
                    borderColor: "rgba(0, 188, 212, 1)",
                    borderWidth: 8,
                    type: 'scatter',
                    pointRadius: 0,
                    yAxisID: 'y',
                  };
                }

                const inputChartIndex = chartData.datasets[0].data.length;
                const newInputData = {
                  x: inputStartRelative,
                  y: "Input Speech",
                  x2: inputEndRelative,
                  index: inputChartIndex
                };
                chartData.datasets[0].data.push(newInputData);
                console.log(`[Chart Debug] Pushed input data: ${JSON.stringify(newInputData)}`);

                const outputChartIndex = chartData.datasets[1].data.length;
                chartData.datasets[1].data.push({
                  x: outputStartRelative,
                  y: "Translated Speech",
                  x2: outputStartRelative, // Initial x2 is same as x, will be updated on playback end
                  index: outputChartIndex,
                  isFinal: data.is_final // Store is_final flag
                });

                currentOutputChartIndex = outputChartIndex; // Set for the current segment

                const currentChartMaxX = latencyChart.options.scales.x.max;
                const latestEndTime = Math.max(inputEndRelative, outputStartRelative);

                if (latestEndTime + 10 > currentChartMaxX) {
                  latencyChart.options.scales.x.max = currentTimeRelative + 10;
                }

                latencyChart.options.scales.x.min = 0;
                
                latencyChart.update();
              }

              // Only reset lastInputTimestamp if this is a final segment
              if (data.is_final) {
                lastInputTimestamp = 0;
                currentOutputChartIndex = -1; // Reset after a final segment is processed
              }
            }
          } else if (data.type === "status") {
            appendLog(data.message || "", "info");
            if (isRecordingModal) {
              modalRecordingStatus.textContent = data.message;
            }
          } else if (data.status === "error") {
            console.error("Server Error:", data.message);
            appendLog(`Server Error: ${data.message}`, "error");
            if (settingsStatusText)
              settingsStatusText.textContent = `Error: ${data.message}`;
            stopRecordingMain();
            stopRecordingModal();
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
      stopRecordingMain();
      stopRecordingModal();
      recordVoiceBtn.disabled = true;
    };

    websocket.onerror = (error) => {
      console.error("WebSocket Error:", error);
      settingsStatusText.textContent = "WebSocket connection error.";
      updateStatus("error", "Error");
      appendLog(`WebSocket Error: ${error.message}`, "error");
      stopRecordingMain();
      stopRecordingModal();
      recordVoiceBtn.disabled = true;
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

  // --- Audio Recording Handling (Main Pipeline) ---
  async function startRecordingMain() {
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
        stopRecordingMain();
        return;
      }

      source.connect(audioProcessor);
      // audioProcessor -> destination is required for some browsers to keep the processing alive
      // audioProcessor.connect(audioContext.destination); // Removed to prevent echo

      isRecording = true;
      updateStatus("on", "Listening...");
      recordVoiceBtn.disabled = true; // Disable record voice button while main recording is active
      startStopBtn.textContent = "Stop";
      startStopBtn.classList.remove("btn-primary");
      startStopBtn.classList.add("btn-stop");
      updateStateIcon(stateListening, true);

      if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({ type: "start" }));
      }
    } catch (error) {
      console.error("Error accessing microphone:", error);
      if (settingsStatusText)
        settingsStatusText.textContent = `Microphone access denied: ${error.message}`;
      updateStatus("error", "Mic Error");
      stopRecordingMain();
    }
  }

  function stopRecordingMain() {
    if (!isRecording) return;

    isRecording = false;
    updateStatus("off", "Ready");
    recordVoiceBtn.disabled = true; // Keep record voice button disabled until re-initialized
    startStopBtn.textContent = "Start";
    startStopBtn.classList.remove("btn-stop");
    startStopBtn.classList.add("btn-primary");
    startStopBtn.disabled = true; // Disable Start/Stop until re-initialized
    updateStateIcon(stateListening, false);
    updateStateIcon(stateTranslating, false);
    updateStateIcon(stateSpeaking, false);
    
    isInitialized = false; // Set isInitialized to false to force re-initialization
    initBtn.disabled = false; // Re-enable init button

    resetUI(); // Call resetUI after updating global states and buttons

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

  // --- Audio Recording Handling (Modal for Voice Training) ---
  let modalAudioChunks = []; // Buffer for audio chunks recorded in modal
  let modalEmaLevel = 0.0; // exponential moving average for smoothing modal mic level

  // Function to set input level for the modal mic bar
  function setInputLevelModal(level) {
    const scaled = Math.sqrt(Math.min(1, Math.max(0, level)));
    modalEmaLevel = modalEmaLevel * (1 - EMA_ALPHA) + scaled * EMA_ALPHA;

    const normalizedLevel = Math.min(1, Math.max(0, modalEmaLevel));
    const activeSegments = Math.ceil(normalizedLevel * NUM_LEVEL_SEGMENTS);

    modalLevelSegments.forEach((segment, index) => {
      if (index < activeSegments) {
        segment.classList.add("active");
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

  async function startRecordingModal() {
    if (isRecordingModal) return;

    modalRecordingStatus.textContent = "Recording...";
    startStopRecordingModalBtn.textContent = "Stop Recording";
    startStopRecordingModalBtn.classList.remove("btn-primary");
    startStopRecordingModalBtn.classList.add("btn-stop");
    agreeToRecordCheckbox.disabled = true;
    uploadSpeakerVoiceModalBtn.disabled = true; // Disable upload during recording
    speakerVoiceUploadModal.disabled = true; // Disable file input during recording
    modalEmaLevel = 0.0; // Reset modal mic level smoothing

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      modalRecordingStatus.textContent = "Microphone not supported by your browser.";
      appendLog("Microphone not supported by browser for modal recording.", "error");
      stopRecordingModal();
      return;
    }

    try {
      let audioConstraints = {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
      };

      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: audioConstraints,
      });

      audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: SAMPLE_RATE,
      });

      if (audioContext.state === "suspended") {
        await audioContext.resume();
      }

      const source = audioContext.createMediaStreamSource(mediaStream);

      try {
        await audioContext.audioWorklet.addModule("ui/audio-processor.js");
        audioProcessor = new AudioWorkletNode(audioContext, "audio-processor");

        modalAudioChunks = []; // Clear previous chunks
        audioProcessor.port.onmessage = (event) => {
          const audioData = event.data;
          modalAudioChunks.push(audioData);

          // Calculate RMS for mic level visualization in modal
          const rms = Math.sqrt(audioData.reduce((sum, val) => sum + val * val, 0) / audioData.length);
          setInputLevelModal(rms);
        };

        source.connect(audioProcessor);
        // audioProcessor.connect(audioContext.destination); // Removed to prevent echo

        isRecordingModal = true;
        appendLog("Started modal voice recording.", "info");

      } catch (workletError) {
        console.error("Error loading AudioWorklet for modal:", workletError);
        modalRecordingStatus.textContent = `Audio processing error: ${workletError.message}`;
        appendLog(`AudioWorklet failed to load for modal: ${workletError.message}`, "error");
        stopRecordingModal();
        return;
      }
    } catch (error) {
      console.error("Error accessing microphone for modal:", error);
      modalRecordingStatus.textContent = `Microphone access denied: ${error.message}`;
      appendLog(`Microphone access denied for modal: ${error.message}`, "error");
      stopRecordingModal();
    }
  }

  async function stopRecordingModal() {
    if (!isRecordingModal) return;

    isRecordingModal = false;
    modalRecordingStatus.textContent = "Processing recorded voice...";
    startStopRecordingModalBtn.textContent = "Start Recording";
    startStopRecordingModalBtn.classList.remove("btn-stop");
    startStopRecordingModalBtn.classList.add("btn-primary");
    startStopRecordingModalBtn.disabled = !agreeToRecordCheckbox.checked; // Re-evaluate disabled state based on checkbox
    agreeToRecordCheckbox.disabled = false;
    uploadSpeakerVoiceModalBtn.disabled = false; // Re-enable upload after recording
    speakerVoiceUploadModal.disabled = false; // Re-enable file input after recording

    if (mediaStream) {
      mediaStream.getTracks().forEach((track) => track.stop());
      mediaStream = null;
    }
    if (audioProcessor) {
      try {
        audioProcessor.port.onmessage = null;
        audioProcessor.disconnect();
      } catch (_) {
        /* ignore */
      }
      audioProcessor = null;
    }
    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }

    if (modalAudioChunks.length > 0) {
      const audioBlob = new Blob(modalAudioChunks, { type: 'audio/wav' }); // Assuming WAV format
      const audioFile = new File([audioBlob], `recorded_voice_${Date.now()}.wav`, { type: 'audio/wav' });

      const formData = new FormData();
      formData.append("file", audioFile);

      modalRecordingStatus.textContent = `Uploading ${audioFile.name}...`;

      try {
        const response = await fetch(`${API_URL}/upload_speaker_voice`, {
          method: "POST",
          body: formData,
        });
        const data = await response.json();

        if (data.status === "success") {
          tempUploadedFileName = audioFile.name; // Store the temporary name
          uploadedSpeakerVoicePath = `speaker_voices/${audioFile.name}`; // Store the path
          appendLog(`Recorded voice uploaded: ${data.message}`, "success");
          modalRecordingStatus.textContent = `Voice recorded and uploaded: ${audioFile.name}.`;
          
          // Hide voice config modal and show naming modal
          voiceConfigModal.style.display = "none";
          voiceNamingModal.style.display = "block";
          
          // Pre-suggest a name
          const selectedLang = modalInputLanguageSelect.value; // Use modal's selected language for naming
          const timestamp = new Date().toLocaleDateString('en-US', { year: 'numeric', month: '2-digit', day: '2-digit' }).replace(/\//g, '-');
          voiceNameInput.value = `${selectedLang.toUpperCase()}_Voice_${timestamp}`;
          voiceNameError.textContent = ""; // Clear any previous error
          
          // Reset voice config modal UI, but don't re-enable initBtn yet
          // Do NOT call resetModalUI here, as it would hide the voiceNamingModal immediately.
          // The user needs to see the naming modal to retry or cancel.
          // resetModalUI(false); // Pass false to prevent re-enabling initBtn prematurely
        } else {
          appendLog(`Recorded voice upload failed: ${data.message}`, "error");
          modalRecordingStatus.textContent = `Upload failed: ${data.message}`;
          // Re-enable the upload button if it was disabled.
          uploadSpeakerVoiceModalBtn.disabled = false;
          // If upload failed, we should still reset the modal UI to allow retrying or cancelling.
          resetModalUI(true);
        }
      } catch (error) {
        console.error("Error uploading recorded voice:", error);
        appendLog(`Error uploading recorded voice: ${error.message}`, "error");
        modalRecordingStatus.textContent = `Upload error: ${error.message}`;
      }
    } else {
      appendLog("No audio recorded in modal.", "warning");
      modalRecordingStatus.textContent = "No audio recorded.";
    }
  }

  // --- Helpers & Events ---
  // Burger menu functionality
  burgerMenuBtn.addEventListener("click", (event) => {
    event.stopPropagation(); // Prevent this click from immediately closing the sidebar
    sidebar.classList.add("open");
    document.body.classList.add("sidebar-open"); // Add class to body
    fetchStoredVoices(); // Load voices when opening sidebar
  });

  closeSidebarBtn.addEventListener("click", () => {
    sidebar.classList.remove("open");
    document.body.classList.remove("sidebar-open"); // Remove class from body
  });

  // Close sidebar when clicking outside of it
  document.addEventListener("click", (event) => {
    if (sidebar.classList.contains("open") && !sidebar.contains(event.target) && event.target !== burgerMenuBtn) {
      sidebar.classList.remove("open");
      document.body.classList.remove("sidebar-open"); // Remove class from body
    }
  });

  // Voice Configuration Modal functionality
  recordVoiceBtn.addEventListener("click", () => {
    voiceConfigModal.style.display = "block";
    updateReadingText(); // Set initial reading text
  });

  closeModalBtn.addEventListener("click", () => {
    voiceConfigModal.style.display = "none";
    stopRecordingModal(); // Ensure recording stops if modal is closed
    resetModalUI();
  });

  cancelModalBtn.addEventListener("click", () => {
    voiceConfigModal.style.display = "none";
    stopRecordingModal(); // Ensure recording stops if modal is cancelled
    resetModalUI();
  });

  agreeToRecordCheckbox.addEventListener("change", () => {
    startStopRecordingModalBtn.disabled = !agreeToRecordCheckbox.checked;
  });

  startStopRecordingModalBtn.addEventListener("click", () => {
    if (isRecordingModal) {
      stopRecordingModal();
    } else {
      startRecordingModal();
    }
  });

  // Event listener for speaker voice file input within modal
  speakerVoiceUploadModal.addEventListener("change", () => {
    if (speakerVoiceUploadModal.files.length > 0) {
      uploadSpeakerVoiceModalBtn.disabled = false;
      currentSpeakerVoiceModal.textContent = speakerVoiceUploadModal.files[0].name;
    } else {
      uploadSpeakerVoiceModalBtn.disabled = true;
      currentSpeakerVoiceModal.textContent = "No voice selected";
    }
  });

  // Event listener for upload speaker voice button within modal
  uploadSpeakerVoiceModalBtn.addEventListener("click", async () => {
    if (speakerVoiceUploadModal.files.length === 0) {
      appendLog("No speaker voice file selected for modal upload.", "warning");
      return;
    }

    const file = speakerVoiceUploadModal.files[0];
    const formData = new FormData();
    formData.append("file", file);

    modalRecordingStatus.textContent = `Uploading ${file.name}...`;
    uploadSpeakerVoiceModalBtn.disabled = true; // Disable during upload

    try {
      const response = await fetch(`${API_URL}/upload_speaker_voice`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      if (data.status === "success") {
        tempUploadedFileName = file.name; // Store the temporary name
        uploadedSpeakerVoicePath = `speaker_voices/${file.name}`; // Store the path
        appendLog(`Speaker voice uploaded: ${data.message}`, "success");
        modalRecordingStatus.textContent = `Voice uploaded: ${file.name}.`;
        
        // Hide voice config modal and show naming modal
        voiceConfigModal.style.display = "none";
        voiceNamingModal.style.display = "block";
        
        // Pre-suggest a name
        const selectedLang = modalInputLanguageSelect.value; // Use modal's selected language for naming
        const timestamp = new Date().toLocaleDateString('en-US', { year: 'numeric', month: '2-digit', day: '2-digit' }).replace(/\//g, '-');
        voiceNameInput.value = `${selectedLang.toUpperCase()}_Voice_${timestamp}`;
        voiceNameError.textContent = ""; // Clear any previous error
        
        // Reset voice config modal UI, but don't re-enable initBtn yet
        // Do NOT call resetModalUI here, as it would hide the voiceNamingModal immediately.
        // The user needs to see the naming modal to retry or cancel.
        // resetModalUI(false); // Pass false to prevent re-enabling initBtn prematurely
      } else {
        appendLog(`Speaker voice upload failed: ${data.message}`, "error");
        modalRecordingStatus.textContent = `Upload failed: ${data.message}`;
        uploadSpeakerVoiceModalBtn.disabled = false; // Re-enable on failure
        // If upload failed, we should still reset the modal UI to allow retrying or cancelling.
        resetModalUI(true);
      }
    } catch (error) {
      console.error("Error uploading speaker voice from modal:", error);
      appendLog(`Error uploading speaker voice from modal: ${error.message}`, "error");
      modalRecordingStatus.textContent = `Upload error: ${error.message}`;
      uploadSpeakerVoiceModalBtn.disabled = false; // Re-enable on failure
      resetModalUI(true); // Reset fully on upload failure
    }
  });

  // Function to update the reading text in the modal based on the modal's language selection
  const baseReadingStatement = "Reading this statement, I agree to provide my voice for cloning purposes. My voice will be used to synthesize translated speech within this application.";

  async function updateReadingText() {
    const selectedLang = modalInputLanguageSelect.value; // Use modal's language select
    if (selectedLang === "en") { // No need to translate if English
      readingText.textContent = baseReadingStatement;
      return;
    }

    // If the main input language is "auto", we should still try to translate to the modal's selected language.
    // The backend's /translate_phrase endpoint expects a source_lang (which is hardcoded to "en" in app.py).
    // So, we only need to ensure the target_lang is correct.

    try {
      const response = await fetch(`${API_URL}/translate_phrase`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          phrase: baseReadingStatement,
          target_lang: selectedLang,
        }),
      });
        const data = await response.json();

        if (data.status === "success") {
          readingText.textContent = data.translated_text;
          appendLog(`Reading text translated to ${selectedLang.toUpperCase()}.`, "info");
        } else {
          readingText.textContent = baseReadingStatement; // Fallback to English
          appendLog(`Failed to translate reading text to ${selectedLang.toUpperCase()}: ${data.message}`, "error");
          // Add specific UI feedback for model loading failure
          if (data.message.includes("MT model for en-") && data.message.includes("not initialized")) {
            modalRecordingStatus.textContent = `Error: MT model for ${selectedLang.toUpperCase()} not found. Please convert it.`;
          }
        }
    } catch (error) {
      console.error("Error translating reading text:", error);
      readingText.textContent = baseReadingStatement; // Fallback to English
      appendLog(`Error translating reading text: ${error.message}`, "error");
    }
  }

  // Initial pipeline initialization (now triggered by a dedicated button)
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

      let initializeUrl = `${API_URL}/initialize?source_lang=${sourceLang}&target_lang=${targetLang}&tts_model_choice=${ttsModel}`;
      if (ttsModel === "xtts" && uploadedSpeakerVoicePath) {
        initializeUrl += `&speaker_wav_path=${encodeURIComponent(uploadedSpeakerVoicePath)}`;
      }

      const response = await fetch(initializeUrl, { method: "POST" });
      const data = await response.json();

      if (data.status === "success") {
        isInitialized = true;
        recordVoiceBtn.disabled = false; // Enable record voice button after initialization
        startStopBtn.disabled = false; // Enable Start/Stop button after initialization
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
        startStopBtn.disabled = true; // Keep Start/Stop button disabled on init failure
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

  // Start/Stop button functionality
  startStopBtn.addEventListener("click", () => {
    if (isRecording) {
      stopRecordingMain();
    } else {
      startRecordingMain();
    }
  });

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
    // Synchronize modal's language select with main input language
    modalInputLanguageSelect.value = event.target.value;
    updateReadingText(); // Update reading text when input language changes
  });

  modalInputLanguageSelect.addEventListener("change", () => {
    updateReadingText(); // Update reading text when modal's language changes
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
    // The upload button logic is now handled within the modal
  });

  // Removed old speakerVoiceUpload and uploadSpeakerVoiceBtn event listeners

  // Function to reset modal UI state
  function resetModalUI(fullReset = true) {
    agreeToRecordCheckbox.checked = false;
    startStopRecordingModalBtn.disabled = true;
    startStopRecordingModalBtn.textContent = "Start Recording";
    startStopRecordingModalBtn.classList.remove("btn-stop");
    startStopRecordingModalBtn.classList.add("btn-primary");
    modalRecordingStatus.textContent = "Ready to record.";
    speakerVoiceUploadModal.value = "";
    uploadSpeakerVoiceModalBtn.disabled = true;
    currentSpeakerVoiceModal.textContent = "No voice selected";
    uploadedSpeakerVoicePath = null; // Clear uploaded path
    modalEmaLevel = 0.0; // Reset modal mic level smoothing
    setInputLevelModal(0); // Reset modal mic level bar

    // Reset naming modal elements
    voiceNameInput.value = "";
    voiceNameError.textContent = "";
    voiceNamingModal.style.display = "none";

    if (fullReset) {
      initBtn.disabled = false; // Re-enable init button on full UI reset
    }
  }

  // Function to fetch and display stored voices in the sidebar
  async function fetchStoredVoices() {
    voiceList.innerHTML = ""; // Clear existing list
    try {
      const response = await fetch(`${API_URL}/list_speaker_voices`);
      const data = await response.json();

      if (data.status === "success" && data.voices.length > 0) {
        data.voices.forEach(voice => {
          const listItem = document.createElement("li");
          listItem.classList.add("voice-item");

          const voiceNameSpan = document.createElement("span");
          voiceNameSpan.classList.add("voice-item-name");
          voiceNameSpan.textContent = voice.replace(".wav", ""); // Display name without .wav
          voiceNameSpan.addEventListener("click", () => { // Keep existing click functionality for selecting voice
            uploadedSpeakerVoicePath = `speaker_voices/${voice}`;
            currentSpeakerVoiceModal.textContent = voice; // Update modal display
            appendLog(`Selected stored voice: ${voice}`, "info");
            sidebar.classList.remove("open"); // Close sidebar after selection
            document.body.classList.remove("sidebar-open"); // Ensure burger menu is visible again
            initBtn.disabled = false; // Re-enable init to apply new voice
            // Also update the TTS model select to XTTS if a voice is selected
            ttsModelSelect.value = "xtts";
            sendConfigUpdate(); // Send config update to backend
          });
          listItem.appendChild(voiceNameSpan);

          const actionsDiv = document.createElement("div");
          actionsDiv.classList.add("voice-item-actions");

          const editBtn = document.createElement("button");
          editBtn.classList.add("voice-action-btn", "edit");
          editBtn.innerHTML = "✏️"; // Edit icon
          editBtn.title = `Rename ${voice.replace(".wav", "")}`;
          editBtn.addEventListener("click", (event) => {
            event.stopPropagation(); // Prevent triggering parent li click
            // Show naming modal for editing
            voiceConfigModal.style.display = "none"; // Hide voice config modal if open
            voiceNamingModal.style.display = "block";
            voiceNameInput.value = voice.replace(".wav", ""); // Pre-fill with current name
            tempUploadedFileName = voice; // Store original name for renaming
            voiceNameError.textContent = ""; // Clear any previous error
          });
          actionsDiv.appendChild(editBtn);

          const deleteBtn = document.createElement("button");
          deleteBtn.classList.add("voice-action-btn", "delete");
          deleteBtn.innerHTML = "🗑️"; // Delete icon
          deleteBtn.title = `Delete ${voice.replace(".wav", "")}`;
          deleteBtn.addEventListener("click", (event) => {
            event.stopPropagation(); // Prevent triggering parent li click
            voiceToDelete = voice; // Store the voice to be deleted
            deleteConfirmationText.textContent = `Are you sure you want to delete the voice "${voice.replace(".wav", "")}"?`;
            deleteConfirmationModal.style.display = "block"; // Show custom confirmation modal
          });
          actionsDiv.appendChild(deleteBtn);

          listItem.appendChild(actionsDiv);
          voiceList.appendChild(listItem);
        });
      } else {
        const listItem = document.createElement("li");
        listItem.textContent = "No voices stored.";
        voiceList.appendChild(listItem);
      }
    } catch (error) {
      console.error("Error fetching stored voices:", error);
      appendLog(`Error fetching stored voices: ${error.message}`, "error");
    }
  }

  // Helper function to escape HTML for safe display
  function escapeHtml(unsafe) {
    return unsafe
      .replace(/&/g, "&")
      .replace(/</g, "<")
      .replace(/>/g, ">")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Initial UI setup
  resetUI(); // Call with default true for full reset
  updateStatus("off", "Awaiting Initialization");
  initBtn.disabled = false; // Init button is enabled by default
  recordVoiceBtn.disabled = true; // Record voice button is disabled until pipeline is initialized
  startStopBtn.disabled = true; // Start/Stop button is disabled until pipeline is initialized
  // burgerMenuBtn.style.display = "block"; // Burger menu is now always visible via CSS

  // Ensure voiceNamingModal is hidden on initial load
  voiceNamingModal.style.display = "none";

  // Voice Naming Modal Event Listeners
  closeNamingModalBtn.addEventListener("click", () => {
    voiceNamingModal.style.display = "none";
    // If user closes naming modal without saving, consider it a full reset
    resetModalUI(true);
  });

  cancelVoiceNameBtn.addEventListener("click", () => {
    voiceNamingModal.style.display = "none";
    // If user cancels naming, consider it a full reset
    resetModalUI(true);
  });

  saveVoiceNameBtn.addEventListener("click", async () => {
    const newVoiceName = voiceNameInput.value.trim();
    if (!newVoiceName) {
      voiceNameError.textContent = "Voice name cannot be empty.";
      return;
    }
    // Basic validation: no special characters or spaces, only alphanumeric and underscores
    if (!/^[a-zA-Z0-9_]+$/.test(newVoiceName)) {
      voiceNameError.textContent = "Only alphanumeric characters and underscores are allowed.";
      return;
    }

    // Assuming tempUploadedFileName is set after successful upload
    if (!tempUploadedFileName) {
      voiceNameError.textContent = "No voice file to rename.";
      return;
    }

    try {
      const response = await fetch(`${API_URL}/rename_speaker_voice`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          old_name: tempUploadedFileName,
          new_name: `${newVoiceName}.wav`, // Ensure .wav extension
        }),
      });
      const data = await response.json();

      if (data.status === "success") {
        appendLog(`Voice renamed to: ${newVoiceName}.wav`, "success");
        uploadedSpeakerVoicePath = `speaker_voices/${newVoiceName}.wav`; // Update path
        initBtn.disabled = false; // Re-enable init button
        voiceNamingModal.style.display = "none";
        resetModalUI(); // Full reset after successful naming
        fetchStoredVoices(); // Refresh sidebar list
      } else {
        voiceNameError.textContent = `Rename failed: ${data.message}`;
        appendLog(`Voice rename failed: ${data.message}`, "error");
      }
    } catch (error) {
      console.error("Error renaming voice:", error);
      voiceNameError.textContent = `Rename error: ${error.message}`;
      appendLog(`Error renaming voice: ${error.message}`, "error");
    }
  });

  // Suppress voice recording modal on initial load
  if (!hasShownVoiceConfigModal) {
    voiceConfigModal.style.display = "none";
  }

  // Ensure deleteConfirmationModal is hidden on initial load
  deleteConfirmationModal.style.display = "none";

  // Delete Confirmation Modal Event Listeners
  closeDeleteModalBtn.addEventListener("click", () => {
    deleteConfirmationModal.style.display = "none";
    voiceToDelete = null; // Clear the voice to delete
  });

  cancelDeleteBtn.addEventListener("click", () => {
    deleteConfirmationModal.style.display = "none";
    voiceToDelete = null; // Clear the voice to delete
  });

  confirmDeleteBtn.addEventListener("click", async () => {
    if (!voiceToDelete) return;

    try {
      const response = await fetch(`${API_URL}/delete_speaker_voice`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ filename: voiceToDelete }),
      });
      const data = await response.json();

      if (data.status === "success") {
        appendLog(`Voice "${voiceToDelete}" deleted successfully.`, "success");
        fetchStoredVoices(); // Refresh list
        // If the deleted voice was the currently selected one, reset selection
        if (uploadedSpeakerVoicePath === `speaker_voices/${voiceToDelete}`) {
          uploadedSpeakerVoicePath = null;
          currentSpeakerVoiceModal.textContent = "No voice selected";
          initBtn.disabled = false; // Re-enable init
          ttsModelSelect.value = "piper"; // Default to piper
          sendConfigUpdate();
        }
      } else {
        appendLog(`Failed to delete voice "${voiceToDelete}": ${data.message}`, "error");
      }
    } catch (error) {
      console.error("Error deleting voice:", error);
      appendLog(`Error deleting voice "${voiceToDelete}": ${error.message}`, "error");
    } finally {
      deleteConfirmationModal.style.display = "none";
      voiceToDelete = null; // Clear the voice to delete
    }
  });

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
      ctx.strokeStyle = 'rgba(255, 235, 59, 0.8)'; // Yellow (log-warning-color)
      ctx.lineWidth = 3; // Thicker line

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

          dataset.data.forEach((dataPoint, dataIndex) => { // Use dataIndex from forEach for robustness
            const yOffset = (dataIndex % 2) * offsetPerSegment; // Alternate offset for better visibility
            const yCenter = base_yCenter + yOffset;

            const startX = x.getPixelForValue(dataPoint.x);
            const endX = x.getPixelForValue(dataPoint.x2);

            console.log(`[Segment Drawing Debug] Dataset: ${dataset.label}, Data Index: ${dataIndex},
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

    // Update the real-time line and X-axis every 100ms for smoother movement and continuous expansion
    setInterval(() => {
      if (isRecording && recordingStartTime > 0 && latencyChart) {
        const currentTimeRelative = (performance.now() - recordingStartTime) / 1000;
        const currentChartMaxX = latencyChart.options.scales.x.max;

        // Continuously expand max X-axis with a buffer
        if (currentTimeRelative + 10 > currentChartMaxX) { // Add 10 seconds buffer
          latencyChart.options.scales.x.max = currentTimeRelative + 10;
        }
        // Ensure min X-axis remains at 0 for continuous timeline
        latencyChart.options.scales.x.min = 0;

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
