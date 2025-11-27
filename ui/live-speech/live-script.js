document.addEventListener('DOMContentLoaded', () => {
    const burgerMenuBtn = document.getElementById('burgerMenuBtn');
    const sidebar = document.getElementById('sidebar');
    const closeSidebarBtn = document.getElementById('closeSidebarBtn');
    const ttsModelSelect = document.getElementById('ttsModelSelect');
    const voiceSelectionGroup = document.getElementById('voice-selection-group');
    const speakerVoiceSelect = document.getElementById('speakerVoiceSelect');
    const mainControlsTopRow = document.querySelector('.main-controls .top-row');
    const recordVoiceBtn = document.getElementById('recordVoiceBtn');
    const voiceConfigModal = document.getElementById('voiceConfigModal');
    const closeModalBtn = document.getElementById('closeModalBtn');
    const agreeToRecordCheckbox = document.getElementById('agreeToRecordCheckbox');
    const startRecordingModalBtn = document.getElementById('startRecordingModalBtn');
    const stopRecordingModalBtn = document.getElementById('stopRecordingModalBtn');
    const uploadSpeakerVoiceModalBtn = document.getElementById('uploadSpeakerVoiceModalBtn');
    const speakerVoiceUploadModal = document.getElementById('speakerVoiceUploadModal');
    const currentSpeakerVoiceModal = document.getElementById('currentSpeakerVoiceModal');
    const modalInputLanguageSelect = document.getElementById('modalInputLanguageSelect');
    const readingText = document.getElementById('readingText');
    const voiceNamingModal = document.getElementById('voiceNamingModal');
    const closeNamingModalBtn = document.getElementById('closeNamingModalBtn');
    const saveVoiceNameBtn = document.getElementById('saveVoiceNameBtn');
    const cancelVoiceNameBtn = document.getElementById('cancelVoiceNameBtn');
    const deleteConfirmationModal = document = document.getElementById('deleteConfirmationModal');
    const closeDeleteModalBtn = document.getElementById('closeDeleteModalBtn');
    const confirmDeleteBtn = document.getElementById('confirmDeleteBtn');
    const cancelDeleteBtn = document.getElementById('cancelDeleteBtn');
    const inputLevelSegments = document.getElementById('inputLevelSegments');
    const modalInputLevelSegments = document.getElementById('modalInputLevelSegments');
    const voiceNameInput = document.getElementById('voiceNameInput'); // Added for voice naming modal

    const SPEAKER_VOICES_DIR = "speaker_voices"; // Define SPEAKER_VOICES_DIR for frontend use

    let mediaRecorder;
    let audioChunks = [];
    let audioStream; // To store the microphone stream
    let currentRecordedBlob = null; // To store the recorded audio blob temporarily
    let voiceToEdit = null; // To store the voice data when editing
    let notificationTimeout; // For custom notifications
    let websocket; // Declare websocket globally
    let latencyChart = null; // Declare latencyChart globally within the DOMContentLoaded scope

    // Global variables for main audio pipeline
    let mainAudioContext = null;
    let mainAnalyser = null;
    let mainMicrophone = null;
    let mainGainNode = null; // Declare mainGainNode globally
    let mainAudioWorkletNode = null; // Replaced ScriptProcessorNode
    let savedAudioConstraints = null; // Store constraints for stream restart

    // --- Input level smoothing state (matching BP xtts) ---
    let emaLevel = 0.0; // exponential moving average for smoothing
    const EMA_ALPHA = 0.3; // Smoothing factor
    const NUM_LEVEL_SEGMENTS = 10; // Number of mic level segments
    let mainAudioStream = null; // To store the microphone stream for the main pipeline
    let chartIdleInterval = null; // Interval for updating chart during idle state
    let isTTSPlaying = false; // Flag to track if TTS is currently playing
    let sessionStartTime = 0; // Track session start time for relative chart timing

    // --- Throttle state for debug logs (avoid spamming console) ---
    let lastLogAt = 0;
    const LOG_THROTTLE_MS = 1000; // log at most once per second

    // Custom Notification Function
    function showNotification(message, type = 'info') {
        // Implementation for notifications
    }

    // Audio output device management
    const audioOutputSelect = document.getElementById('audioOutputSelect');
    const virtualMicHint = document.getElementById('virtualMicHint');
    let selectedAudioOutputDeviceId = null;

    // Enumerate and populate audio output devices
    async function enumerateAudioOutputDevices() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const audioOutputs = devices.filter(device => device.kind === 'audiooutput');

            // Clear existing options except default
            audioOutputSelect.innerHTML = '<option value="">Default</option>';

            let virtualDeviceFound = false;
            let virtualDeviceName = null;

            audioOutputs.forEach(device => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.textContent = device.label || `Audio Output ${audioOutputs.indexOf(device) + 1}`;
                audioOutputSelect.appendChild(option);

                // Auto-detect virtual device (BlackHole or VB-Cable)
                const label = device.label.toLowerCase();
                if (label.includes('blackhole') || label.includes('vb-cable') || label.includes('vb cable')) {
                    option.selected = true;
                    selectedAudioOutputDeviceId = device.deviceId;
                    virtualDeviceFound = true;
                    virtualDeviceName = device.label;
                    console.log(`Frontend: Auto-selected virtual audio device: ${device.label}`);
                }
            });

            // Update hint text based on detection
            if (virtualDeviceFound) {
                virtualMicHint.textContent = `✓ Virtual device detected: ${virtualDeviceName}`;
                virtualMicHint.style.color = 'var(--success-color, #4caf50)';
                showNotification(`Audio will route to: ${virtualDeviceName}`, 'success');
            } else {
                virtualMicHint.textContent = '⚠ No virtual device found. Install BlackHole or VB-Cable.';
                virtualMicHint.style.color = 'var(--warning-color, #ff9800)';
                console.warn('Frontend: No virtual audio device detected. Translation will play through default output.');
            }

            // Store selection in localStorage
            if (selectedAudioOutputDeviceId) {
                localStorage.setItem('preferredAudioOutput', selectedAudioOutputDeviceId);
            }

        } catch (error) {
            console.error('Frontend: Error enumerating audio devices:', error);
            virtualMicHint.textContent = '❌ Could not access audio devices';
            virtualMicHint.style.color = 'var(--danger-color, #f44336)';
        }
    }

    // Handle audio output device selection change
    if (audioOutputSelect) {
        audioOutputSelect.addEventListener('change', (event) => {
            const deviceId = event.target.value;
            selectedAudioOutputDeviceId = deviceId || null;

            // Save to localStorage
            if (deviceId) {
                localStorage.setItem('preferredAudioOutput', deviceId);
                const selectedLabel = event.target.options[event.target.selectedIndex].textContent;
                console.log(`Frontend: Audio output changed to: ${selectedLabel}`);
                showNotification(`Audio output: ${selectedLabel}`, 'info');
            } else {
                localStorage.removeItem('preferredAudioOutput');
                console.log('Frontend: Audio output set to default');
                showNotification('Audio output: Default', 'info');
            }
        });
    }

    // Enumerate devices on page load
    enumerateAudioOutputDevices();

    // Re-enumerate when devices change (e.g., virtual device installed/removed)
    navigator.mediaDevices.addEventListener('devicechange', () => {
        console.log('Frontend: Audio devices changed, re-enumerating...');
        enumerateAudioOutputDevices();
    });

    function showNotification(message, type = 'info') {
        const notificationBar = document.getElementById('notificationBar');
        if (!notificationBar) {
            console.warn('Notification bar element not found.');
            // Fallback to console log if notification bar doesn't exist, no alert
            console.log(`Notification (${type}): ${message}`);
            return;
        }

        notificationBar.textContent = message;
        notificationBar.className = `notification-bar ${type} show`; // Reset classes and show

        if (notificationTimeout) {
            clearTimeout(notificationTimeout);
        }
        notificationTimeout = setTimeout(() => {
            notificationBar.classList.remove('show');
        }, 5000); // Hide after 5 seconds
    }

    // Function to update modal mic input level
    function updateModalMicLevel(level) {
        if (modalInputLevelSegments) {
            Array.from(modalInputLevelSegments.children).forEach((segment, index) => {
                if (index < level) {
                    segment.classList.add('active');
                } else {
                    segment.classList.remove('active');
                }
            });
        }
    }

    // Function to update main UI mic input level (matching BP xtts implementation)
    function updateMainMicLevel(level) {
        const segments = inputLevelSegments.querySelectorAll('.segment');
        if (!segments || segments.length === 0) {
            console.warn('Frontend: Mic level segments not found');
            return;
        }

        // Expect 'level' to be a 0..1 RMS-like magnitude from backend
        // Apply sqrt to approximate perceptual loudness, then EMA smoothing
        const scaled = Math.sqrt(Math.min(1, Math.max(0, level)));
        emaLevel = emaLevel * (1 - EMA_ALPHA) + scaled * EMA_ALPHA;

        const normalizedLevel = Math.min(1, Math.max(0, emaLevel));
        const activeSegments = Math.ceil(normalizedLevel * NUM_LEVEL_SEGMENTS);

        // Debug logging - remove after testing
        if (level > 0.001 || activeSegments > 0) {
            console.log(`[MIC] RMS: ${level.toFixed(4)}, Scaled: ${scaled.toFixed(4)}, EMA: ${emaLevel.toFixed(4)}, Segments: ${activeSegments}/${NUM_LEVEL_SEGMENTS}`);
        }

        segments.forEach((segment, index) => {
            if (index < activeSegments) {
                segment.classList.add('active');

                // Style classes for medium/high ranges
                if (normalizedLevel > 0.7) {
                    segment.classList.add('high');
                    segment.classList.remove('medium');
                } else if (normalizedLevel > 0.3) {
                    segment.classList.add('medium');
                    segment.classList.remove('high');
                } else {
                    segment.classList.remove('medium', 'high');
                }
            } else {
                segment.classList.remove('active', 'medium', 'high');
            }
        });
    }

    // WebSocket setup
    function setupWebSocket() {
        const WS_URL = `wss://${window.location.host}/ws`; // Adjust if your backend is on a different host/port
        websocket = new WebSocket(WS_URL);

        websocket.onopen = (event) => {
            console.log('WebSocket opened:', event);
            showNotification('Connected to translation service.', 'success');
        };

        websocket.onmessage = async (event) => { // Made async to handle Blob reading
            if (typeof event.data === 'string') {
                const message = JSON.parse(event.data);
                if (message.type === 'audio_level') {
                    // Backend sends RMS values (0-1 range)
                    // Pass raw RMS to updateMainMicLevel for EMA smoothing and sqrt scaling
                    const rms = message.level || 0;
                    updateMainMicLevel(rms);
                } else if (message.type === 'models_loading_status') {
                    if (message.fully_loaded) {
                        initBtn.disabled = true; // Disable initialize button
                        initBtn.classList.remove('btn-active'); // Dim initialize button
                        startStopBtn.disabled = false; // Enable start button
                        startStopBtn.classList.add('btn-active'); // Light up start button
                        statusLabel.textContent = 'Ready';
                        statusIndicator.classList.remove('off');
                        statusIndicator.classList.add('on');
                        showNotification('Models are fully loaded and ready!', 'success');
                    } else {
                        initBtn.disabled = false; // Enable initialize button
                        initBtn.classList.add('btn-active'); // Light up initialize button (if it was dimmed)
                        startStopBtn.disabled = true; // Disable start button
                        startStopBtn.classList.remove('btn-active'); // Dim start button
                        statusLabel.textContent = 'Error';
                        statusIndicator.classList.remove('on');
                        statusIndicator.classList.add('off');
                        showNotification('Model loading failed or is incomplete.', 'error');
                    }
                } else if (message.type === 'transcription_result') {
                    document.getElementById('transcriptionOutput').textContent = message.transcribed;
                } else if (message.type === 'translation_result') {
                    document.getElementById('translationOutput').textContent = message.translated;
                } else if (message.type === 'final_metrics') {
                    console.log('Frontend: Final Metrics:', message.metrics);
                    if (latencyChart) {
                        const currentTime = performance.now() / 1000; // Current time in seconds
                        const relativeTime = currentTime - sessionStartTime;
                        const inputActive = message.metrics.stt_time > 0; // If STT happened, input was active
                        const translationActive = message.metrics.tts_time > 0; // If TTS happened, translation was active
                        updateLatencyChart(relativeTime, inputActive, translationActive);
                        console.log(`Frontend: Chart updated at ${relativeTime.toFixed(2)}s. Input Active: ${inputActive}, Translation Active: ${translationActive}`);
                    } else {
                        console.warn('Frontend: Latency chart not initialized when receiving final_metrics.');
                    }
                } else if (message.type === 'status' || message.type === 'notification') {
                    showNotification(message.message, message.type === 'notification' ? message.type : 'info');
                } else if (message.type === 'mt_conversion_status') {
                    if (message.status === 'started') {
                        showNotification(`Starting MT model conversion for ${message.model_name}... This may take a moment.`, 'info');
                    } else if (message.status === 'completed') {
                        showNotification(`MT model conversion for ${message.model_name} completed.`, 'success');
                    } else if (message.status === 'failed') {
                        showNotification(`MT model conversion for ${message.model_name} failed: ${message.error}`, 'error');
                    }
                }
            } else if (event.data instanceof Blob) {
                // Handle Blob (audio data)
                const audioBlob = event.data;
                const audioUrl = URL.createObjectURL(audioBlob);
                const audio = new Audio(audioUrl);

                // Auto-route to BlackHole/VB-Cable if detected
                if (selectedAudioOutputDeviceId && typeof audio.setSinkId === 'function') {
                    audio.setSinkId(selectedAudioOutputDeviceId)
                        .then(() => {
                            console.log('Frontend: Audio routed to virtual device');
                        })
                        .catch(err => {
                            console.error('Frontend: Error setting audio output device:', err);
                        });
                } else {
                    console.log('Frontend: TTS audio playing through DEFAULT speakers (BlackHole routing disabled or not supported)');
                }

                // Play audio
                console.log('Frontend: Starting TTS playback. Stopping audio capture...');
                isTTSPlaying = true; // Set flag

                // CRITICAL: Completely stop audio capture to prevent browser from capturing TTS
                pauseAudioCapture();

                audio.play()
                    .then(() => {
                        console.log('Frontend: Audio playback started.');
                    })
                    .catch(err => {
                        console.error('Frontend: Audio playback error:', err);
                        isTTSPlaying = false;
                        // Resume capture even if playback fails
                        resumeAudioCapture();
                    });

                audio.onended = () => {
                    console.log('Frontend: TTS playback finished. Restarting audio capture...');
                    isTTSPlaying = false;
                    URL.revokeObjectURL(audioUrl);

                    // Restart audio capture after TTS finishes
                    resumeAudioCapture();
                };
                console.log('Frontend: Received and playing translated audio blob.');
            } else if (event.data instanceof ArrayBuffer) {
                // Handle ArrayBuffer if the backend sends it directly
                const audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const audioBuffer = await audioContext.decodeAudioData(event.data);
                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContext.destination);
                source.start();
                console.log('Frontend: Received and playing audio ArrayBuffer.');
            }
        };

        websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            showNotification('WebSocket error. Please check console for details.', 'error');
        };

        websocket.onclose = (event) => {
            console.log('WebSocket closed:', event);
            showNotification('Disconnected from translation service.', 'warning');
            // Attempt to reconnect after a delay
            setTimeout(setupWebSocket, 5000);
        };
    }

    // Call setupWebSocket on page load
    setupWebSocket();

    const startStopBtn = document.getElementById('startStopBtn');
    const statusLabel = document.getElementById('statusLabel');
    const statusIndicator = document.getElementById('statusIndicator');
    const initBtn = document.getElementById('initBtn'); // Get the initialize button

    let isStreaming = false; // Track if real-time streaming is active
    let modalAudioContext = null; // For modal mic level visualization
    let modalAnalyser = null;
    let modalMicrophone = null;
    let animationFrameId = null; // To cancel animation frame for modal mic level

    // Set initial button states
    initBtn.disabled = false;
    startStopBtn.disabled = true;
    initBtn.classList.add('btn-active'); // Initialize button is active by default
    startStopBtn.classList.remove('btn-active');

    // Function to start real-time audio streaming
    async function startRealtimeAudioStreaming() {
        console.log('Frontend: Attempting to start real-time audio streaming...'); // Added log
        try {
            console.log('Frontend: Calling navigator.mediaDevices.getUserMedia...'); // Added log before getUserMedia

            // CRITICAL FIX: Enumerate devices and explicitly exclude BlackHole from mic input
            const devices = await navigator.mediaDevices.enumerateDevices();
            const audioInputs = devices.filter(d => d.kind === 'audioinput');

            // Find a non-BlackHole microphone
            let selectedMicId = null;
            for (const device of audioInputs) {
                const isBlackHole = device.label.toLowerCase().includes('blackhole');
                console.log(`Frontend: Found audio input: ${device.label} (BlackHole: ${isBlackHole})`);

                if (!isBlackHole && !selectedMicId) {
                    selectedMicId = device.deviceId;
                    console.log(`Frontend: Selected mic: ${device.label}`);
                }
            }

            // If no specific device found, use default (but warn if it's BlackHole)
            const audioConstraints = selectedMicId
                ? {
                    audio: {
                        deviceId: { exact: selectedMicId },
                        echoCancellation: false,  // CRITICAL: Disable to prevent capturing TTS from BlackHole
                        noiseSuppression: false,  // Disable to prevent audio artifacts
                        autoGainControl: true     // Keep AGC for consistent mic levels
                    }
                }
                : {
                    audio: {
                        echoCancellation: false,
                        noiseSuppression: false,
                        autoGainControl: true
                    }
                };

            // Save constraints for stream restart after TTS
            savedAudioConstraints = audioConstraints;

            mainAudioStream = await navigator.mediaDevices.getUserMedia(audioConstraints);

            // Verify we're not using BlackHole
            const track = mainAudioStream.getAudioTracks()[0];
            if (track.label.toLowerCase().includes('blackhole')) {
                console.error('Frontend: CRITICAL ERROR - Microphone input is BlackHole! This will cause feedback loop!');
                showNotification('ERROR: Microphone input is BlackHole! This will cause feedback. Please change your system audio settings.', 'error');
                stopRealtimeAudioStreaming();
                return;
            }

            console.log('Frontend: MediaStream obtained. Active:', mainAudioStream.active, 'Tracks:', mainAudioStream.getAudioTracks());
            mainAudioStream.getAudioTracks().forEach((track, index) => {
                console.log(`Frontend: Audio Track ${index}: id=${track.id}, label=${track.label}, readyState=${track.readyState}, enabled=${track.enabled}`);
            });

            // Explicitly set sample rate to 16000 Hz for consistency with backend
            mainAudioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            console.log('Frontend: AudioContext created with sampleRate:', mainAudioContext.sampleRate, 'and state:', mainAudioContext.state);

            // Ensure AudioContext is running, especially after user gesture
            if (mainAudioContext.state === 'suspended') {
                await mainAudioContext.resume();
                console.log('Frontend: AudioContext resumed to state:', mainAudioContext.state);
            }
            await mainAudioContext.audioWorklet.addModule('/ui/live-speech/audio-processor.js'); // Load AudioWorklet module
            mainMicrophone = mainAudioContext.createMediaStreamSource(mainAudioStream);
            mainAnalyser = mainAudioContext.createAnalyser();
            mainGainNode = mainAudioContext.createGain(); // Assign to global mainGainNode
            mainGainNode.gain.value = 4.0; // Increased gain for better mic pickup (was 2.0 = 6dB, now 4.0 = 12dB boost)
            console.log('Frontend: Added GainNode with initial gain:', mainGainNode.gain.value);

            mainAudioWorkletNode = new AudioWorkletNode(mainAudioContext, 'audio-processor'); // Create AudioWorkletNode

            mainAudioWorkletNode.port.onmessage = (event) => {
                const inputBuffer = event.data; // Data is the Float32Array from the AudioWorklet

                // --- DEBUGGING LOGS START ---
                // Check if the inputBuffer contains any non-zero values
                const hasAudioData = inputBuffer.some(value => value !== 0);
                if (!inputBuffer || inputBuffer.length === 0) {
                    console.warn('Frontend: Received empty or null inputBuffer from AudioWorkletProcessor.');
                    return;
                }

                // Check if buffer is all zeros (silence)
                const isAllZeros = inputBuffer.every(sample => sample === 0);
                if (isAllZeros) {
                    // console.warn('Frontend: Received all-zeros buffer from AudioWorkletProcessor (Silence).');
                    // We still send silence to backend so VAD can detect silence
                }

                // MUTE LOGIC: If TTS is playing, do not send audio to backend
                if (isTTSPlaying) {
                    // Optional: Send silence instead of nothing to keep VAD state consistent?
                    // For now, let's just drop the packets to mimic "blocking"
                    // console.log('Frontend: Mic muted during TTS playback.');
                    return;
                }

                // The AudioWorkletNode already outputs Float32Array, which is what the backend expects.
                // Sending the raw Float32Array buffer directly.
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(inputBuffer.buffer); // Send raw Float32Array buffer
                }
            };

            mainMicrophone.connect(mainGainNode); // Connect microphone to GainNode
            mainGainNode.connect(mainAnalyser); // Connect GainNode to Analyser
            mainAnalyser.connect(mainAudioWorkletNode); // Connect analyser to AudioWorkletNode
            // mainAudioWorkletNode.connect(mainAudioContext.destination); // Disconnect AudioWorkletNode from destination to prevent mic loopback

            // Start interval for chart idle state updates (500ms)
            // Start interval for chart idle state updates (500ms)
            // Start interval for chart idle state updates (500ms)
            if (typeof window.updateLatencyChart === 'function') {
                sessionStartTime = performance.now() / 1000; // Set session start time
                chartIdleInterval = setInterval(() => {
                    // Always update chart during idle (no need to check isProcessingSpeech)
                    const currentTime = performance.now() / 1000;
                    const relativeTime = currentTime - sessionStartTime;
                    window.updateLatencyChart(relativeTime, false, false);
                }, 500);
            }

            isStreaming = true;
            startStopBtn.textContent = 'Stop';
            statusLabel.textContent = 'Listening...';
            statusIndicator.classList.remove('off');
            statusIndicator.classList.add('on');
            showNotification('Real-time streaming started.', 'success');
        } catch (err) {
            console.error('Error starting real-time audio streaming:', err);
            showNotification('Failed to start real-time audio streaming. Check microphone access.', 'error');
            stopRealtimeAudioStreaming(); // Ensure cleanup on error
        }
    }

    // Helper function to pause audio capture (stop stream during TTS)
    async function pauseAudioCapture() {
        console.log('Frontend: Pausing audio capture for TTS playback...');

        // Disconnect and stop everything
        if (mainAudioWorkletNode) {
            mainAudioWorkletNode.disconnect();
            mainAudioWorkletNode = null;
        }
        if (mainAnalyser) {
            mainAnalyser.disconnect();
        }
        if (mainGainNode) {
            mainGainNode.disconnect();
        }
        if (mainMicrophone) {
            mainMicrophone.disconnect();
        }
        if (mainAudioStream) {
            mainAudioStream.getTracks().forEach(track => track.stop());
            mainAudioStream = null;
        }

        console.log('Frontend: Audio capture paused.');
    }

    // Helper function to resume audio capture (restart stream after TTS)
    async function resumeAudioCapture() {
        console.log('Frontend: Resuming audio capture after TTS playback...');

        try {
            // Restart the stream with saved constraints
            if (!savedAudioConstraints) {
                console.error('Frontend: No saved audio constraints to resume!');
                return;
            }

            mainAudioStream = await navigator.mediaDevices.getUserMedia(savedAudioConstraints);
            console.log('Frontend: MediaStream restarted.');

            // Recreate the audio processing chain
            mainMicrophone = mainAudioContext.createMediaStreamSource(mainAudioStream);

            // Reconnect GainNode
            mainMicrophone.connect(mainGainNode);
            mainGainNode.connect(mainAnalyser);

            // Recreate AudioWorkletNode
            mainAudioWorkletNode = new AudioWorkletNode(mainAudioContext, 'audio-processor');

            mainAudioWorkletNode.port.onmessage = (event) => {
                const audioData = event.data;
                if (isTTSPlaying) {
                    return; // Extra safety check
                }

                const audioDataFloat32 = new Float32Array(audioData);
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(audioDataFloat32.buffer);
                }
            };

            mainAnalyser.connect(mainAudioWorkletNode);

            console.log('Frontend: Audio capture resumed successfully.');
        } catch (err) {
            console.error('Frontend: Error resuming audio capture:', err);
            showNotification('Failed to resume audio capture. Please restart streaming.', 'error');
        }
    }

    // Function to stop real-time audio streaming
    function stopRealtimeAudioStreaming() {
        if (mainAudioStream) {
            mainAudioStream.getTracks().forEach(track => track.stop());
            mainAudioStream = null;
        }
        if (mainAudioWorkletNode) {
            mainAudioWorkletNode.disconnect();
            mainAudioWorkletNode = null;
        }
        if (mainAnalyser) {
            mainAnalyser.disconnect();
            mainAnalyser = null;
        }
        // Disconnect and nullify the GainNode
        if (mainGainNode) {
            mainGainNode.disconnect();
            mainGainNode = null;
        }
        if (mainMicrophone) {
            mainMicrophone.disconnect();
            mainMicrophone = null;
        }
        if (mainAudioContext) {
            mainAudioContext.close();
            mainAudioContext = null;
        }

        // Clear chart idle interval
        if (chartIdleInterval) {
            clearInterval(chartIdleInterval);
            chartIdleInterval = null;
        }

        // Reset EMA level
        emaLevel = 0;

        isStreaming = false;
        startStopBtn.textContent = 'Start';
        statusLabel.textContent = 'Ready';
        statusIndicator.classList.remove('on');
        statusIndicator.classList.add('off');
        updateMainMicLevel(0); // Reset mic level display
        updateMainMicLevel(0); // Reset mic level display

        // Reset chart if it exists
        if (typeof window.resetLatencyChart === 'function') {
            window.resetLatencyChart();
        }

        showNotification('Real-time streaming stopped.', 'info');
    }

    // Event listener for the main Start/Stop button
    if (startStopBtn) {
        startStopBtn.addEventListener('click', () => {
            console.log('Frontend: Start/Stop button clicked. isStreaming:', isStreaming); // Added log
            if (isStreaming) {
                stopRealtimeAudioStreaming();
            } else {
                startRealtimeAudioStreaming();
            }
        });
    }

    // Function to send VAD config to backend
    async function sendVadConfigToBackend(vadAggressiveness, silenceRmsThreshold) {
        showNotification('Updating VAD configuration...', 'info');
        const userToken = localStorage.getItem('userToken');
        if (!userToken) {
            showNotification('You must be logged in to update VAD settings.', 'error');
            return;
        }

        const sourceLang = inputLanguageSelect.value;
        const targetLang = document.getElementById('outputLanguageSelect').value;
        const ttsModelChoice = ttsModelSelect.value;
        const sttModelSize = "tiny"; // Default STT model size
        const vadEnabled = true; // VAD is always enabled if config is being sent

        let speakerWavPath = null;
        let speakerText = null;
        let speakerLang = null;

        if (ttsModelChoice === 'xtts') {
            const selectedVoiceOption = speakerVoiceSelect.options[speakerVoiceSelect.selectedIndex];
            if (selectedVoiceOption && selectedVoiceOption.value) {
                speakerWavPath = selectedVoiceOption.value;
                speakerLang = selectedVoiceOption.dataset.voiceLang || 'en';
            }
        }

        const queryParams = new URLSearchParams({
            source_lang: sourceLang,
            target_lang: targetLang,
            tts_model_choice: ttsModelChoice
        });

        try {
            const response = await fetch(`/initialize?${queryParams.toString()}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${userToken}`
                },
                body: JSON.stringify({
                    stt_model_size: sttModelSize,
                    vad_enabled_param: vadEnabled,
                    speaker_wav_path: speakerWavPath,
                    speaker_text: speakerText,
                    speaker_lang: speakerLang,
                    vad_aggressiveness: vadAggressiveness, // Send VAD aggressiveness
                    silence_rms_threshold: silenceRmsThreshold // Send silence RMS threshold
                })
            });

            if (response.ok) {
                const result = await response.json();
                showNotification('VAD configuration updated successfully!', 'success');
                console.log('VAD config update result:', result);
            } else {
                let errorData;
                const contentType = response.headers.get('content-type');
                if (contentType && contentType.includes('application/json')) {
                    errorData = await response.json();
                } else {
                    errorData = await response.text();
                    showNotification(`Failed to update VAD config: ${errorData}`, 'error');
                }
                console.error('VAD config update failed:', errorData);
            }
        } catch (error) {
            showNotification('Network error during VAD config update.', 'error');
            console.error('Error during VAD config update:', error);
        }
    }

    // Event listener for the Initialize Pipeline button
    if (initBtn) {
        initBtn.addEventListener('click', async () => {
            showNotification('Initializing models...', 'info');
            const userToken = localStorage.getItem('userToken');
            if (!userToken) {
                showNotification('You must be logged in to initialize models.', 'error');
                return;
            }

            const sourceLang = inputLanguageSelect.value;
            const targetLang = document.getElementById('outputLanguageSelect').value;
            const ttsModelChoice = ttsModelSelect.value;
            const sttModelSize = "tiny"; // Default STT model size
            const vadEnabled = true; // Default VAD enabled state

            let speakerWavPath = null;
            let speakerText = null;
            let speakerLang = null;

            if (ttsModelChoice === 'xtts') {
                const selectedVoiceOption = speakerVoiceSelect.options[speakerVoiceSelect.selectedIndex];
                if (selectedVoiceOption && selectedVoiceOption.value) {
                    speakerWavPath = selectedVoiceOption.value;
                    speakerLang = selectedVoiceOption.dataset.voiceLang || 'en'; // Fallback to 'en'
                } else {
                    showNotification('Please select a voice for XTTS.', 'warning');
                    // Do not return here, allow initialization with null speaker if XTTS is selected but no voice
                }
            }

            // Construct query parameters for source_lang, target_lang, tts_model_choice
            const queryParams = new URLSearchParams({
                source_lang: sourceLang,
                target_lang: targetLang,
                tts_model_choice: ttsModelChoice
            });

            try {
                const response = await fetch(`/initialize?${queryParams.toString()}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${userToken}`
                    },
                    body: JSON.stringify({
                        stt_model_size: sttModelSize,
                        vad_enabled_param: vadEnabled,
                        speaker_wav_path: speakerWavPath,
                        speaker_text: speakerText,
                        speaker_lang: speakerLang
                    })
                });

                if (response.ok) {
                    const result = await response.json();
                    showNotification('Models initialized successfully!', 'success');
                    console.log('Initialization result:', result);
                    // Immediate UI feedback for success
                    initBtn.disabled = true;
                    initBtn.classList.remove('btn-active');
                    startStopBtn.disabled = false;
                    startStopBtn.classList.add('btn-active');
                    statusLabel.textContent = 'Ready';
                    statusIndicator.classList.remove('off');
                    statusIndicator.classList.add('on');
                } else {
                    let errorData;
                    const contentType = response.headers.get('content-type');
                    if (contentType && contentType.includes('application/json')) {
                        errorData = await response.json();
                    } else { // If not JSON, it's plain text
                        errorData = await response.text();
                        showNotification(`Failed to initialize models: ${errorData}`, 'error'); // Display raw text directly
                    }
                    console.error('Initialization failed:', errorData);
                    // Immediate UI feedback for failure
                    initBtn.disabled = false;
                    initBtn.classList.add('btn-active');
                    startStopBtn.disabled = true;
                    startStopBtn.classList.remove('btn-active');
                    statusLabel.textContent = 'Error';
                    statusIndicator.classList.remove('on');
                    statusIndicator.classList.add('off');
                }
            } catch (error) {
                showNotification('Network error during model initialization.', 'error');
                console.error('Error during initialization:', error);
                // Immediate UI feedback for network error
                initBtn.disabled = false;
                initBtn.classList.add('btn-active');
                startStopBtn.disabled = true;
                startStopBtn.classList.remove('btn-active');
                statusLabel.textContent = 'Error';
                statusIndicator.classList.remove('on');
                statusIndicator.classList.add('off');
            }
        });
    }

    // Sidebar toggle
    if (burgerMenuBtn) {
        burgerMenuBtn.addEventListener('click', () => {
            sidebar.classList.add('open');
        });
    }

    if (closeSidebarBtn) {
        closeSidebarBtn.addEventListener('click', () => {
            sidebar.classList.remove('open');
        });
    }

    // Close sidebar when clicking outside (optional, but good UX)
    document.addEventListener('click', (event) => {
        if (!sidebar.contains(event.target) && !burgerMenuBtn.contains(event.target) && sidebar.classList.contains('open')) {
            sidebar.classList.remove('open');
        }
    });

    const voiceLanguageSections = document.getElementById('voiceLanguageSections');
    const accountButton = document.getElementById('accountButton');
    const accountFloatingContainer = document.getElementById('accountFloatingContainer');
    const logoutButton = document.querySelector('.logout-button');
    const accountUserIcon = document.getElementById('accountUserIcon');
    const inputLanguageSelect = document.getElementById('inputLanguageSelect');
    const accountDisplayUsername = document.getElementById('accountDisplayUsername');
    const accountDisplayEmail = document.getElementById('accountDisplayEmail');
    const accountUsernameSpan = document.getElementById('accountUsername'); // Span inside the account button

    // Function to load voices from backend API
    async function loadVoices() {
        const userToken = localStorage.getItem('userToken');
        if (!userToken) {
            console.warn('No user token found. Loading only default voices.');
            // Optionally, redirect to login or show a message
            // For now, we'll proceed without user-specific voices
        }

        try {
            const response = await fetch('/api/voices', {
                headers: {
                    'Authorization': `Bearer ${userToken}`
                }
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            const voices = await response.json();
            return voices;
        } catch (error) {
            console.error('Error loading voices:', error);
            return [];
        }
    }

    // Function to render voices in the sidebar and XTTS dropdown
    async function renderVoices() {
        const voices = await loadVoices();

        // Render in sidebar
        if (voiceLanguageSections) {
            const voicesByLanguage = voices.reduce((acc, voice) => {
                // Ensure voice.language is a string, default to 'unknown' if undefined
                const lang = (voice.language || 'unknown').toLowerCase();
                if (!acc[lang]) {
                    acc[lang] = [];
                }
                acc[lang].push(voice);
                return acc;
            }, {});

            voiceLanguageSections.innerHTML = ''; // Clear existing content

            for (const langCode in voicesByLanguage) {
                const langSection = document.createElement('div');
                langSection.classList.add('voice-language-section');

                const langToggle = document.createElement('h4');
                langToggle.classList.add('language-toggle');
                langToggle.innerHTML = `-${langCode} <span class="material-symbols-outlined expand-icon">expand_more</span>`;
                langSection.appendChild(langToggle);

                const ul = document.createElement('ul');
                ul.classList.add('voice-list', 'hidden'); // Start hidden
                langSection.appendChild(ul);

                voicesByLanguage[langCode].forEach(voice => {
                    const li = document.createElement('li');
                    li.dataset.voicePath = voice.path; // Store voice path
                    li.dataset.voiceText = voice.transcribed_text; // Store transcribed text
                    li.dataset.voiceLang = voice.language; // Store voice language
                    li.innerHTML = `
                        <span>--${voice.name}</span>
                        <div class="voice-actions">
                            <button data-id="${voice.id}" class="edit-voice-btn">Edit</button>
                            <button data-id="${voice.id}" class="delete-voice-btn">Delete</button>
                        </div>
                    `;
                    ul.appendChild(li);
                });
                voiceLanguageSections.appendChild(langSection);
            }

            // Add event listeners for language toggles and voice selection
            voiceLanguageSections.addEventListener('click', (event) => {
                const toggleElement = event.target.closest('.language-toggle');
                if (toggleElement) {
                    const voiceList = toggleElement.nextElementSibling;
                    const expandIcon = toggleElement.querySelector('.expand-icon');

                    voiceList.classList.toggle('hidden');
                    if (voiceList.classList.contains('hidden')) {
                        expandIcon.textContent = 'expand_more';
                    } else {
                        expandIcon.textContent = 'expand_less';
                    }
                } else if (event.target.classList.contains('edit-voice-btn')) {
                    const voiceId = event.target.dataset.id;
                    voiceToEdit = voices.find(v => v.id === voiceId); // Find the full voice object
                    if (voiceToEdit) {
                        voiceNameInput.value = voiceToEdit.name; // Populate input with current name
                        voiceNamingModal.classList.add('open'); // Open naming modal for editing
                    }
                } else if (event.target.classList.contains('delete-voice-btn')) {
                    const voiceId = event.target.dataset.id;
                    voiceToEdit = voices.find(v => v.id === voiceId); // Find the full voice object
                    if (voiceToEdit) {
                        document.getElementById('deleteConfirmationText').textContent = `Are you sure you want to delete "${voiceToEdit.name}"?`;
                        deleteConfirmationModal.classList.add('open'); // Open delete confirmation modal
                    }
                } else if (event.target.closest('li') && !event.target.classList.contains('edit-voice-btn') && !event.target.classList.contains('delete-voice-btn')) {
                    const selectedVoiceLi = event.target.closest('li');
                    const voicePath = selectedVoiceLi.dataset.voicePath;
                    const voiceText = selectedVoiceLi.dataset.voiceText;
                    const voiceLang = selectedVoiceLi.dataset.voiceLang;
                    const voiceName = selectedVoiceLi.querySelector('span').textContent.replace('--', '');

                    // Remove 'selected' class from all other voice list items
                    document.querySelectorAll('.voice-list li').forEach(item => {
                        item.classList.remove('selected');
                    });
                    // Add 'selected' class to the clicked voice
                    selectedVoiceLi.classList.add('selected');

                    // Update speaker voice dropdown
                    const optionExists = Array.from(speakerVoiceSelect.options).some(option => option.value === voicePath);
                    if (!optionExists) {
                        const newOption = document.createElement('option');
                        newOption.value = voice.path;
                        newOption.textContent = voice.name;
                        speakerVoiceSelect.appendChild(newOption);
                    }
                    speakerVoiceSelect.value = voicePath;
                    speakerVoiceSelect.dispatchEvent(new Event('change')); // Trigger change event

                    // Update current selected voice in UI
                    console.log(`Selected voice: ${voiceName} (Path: ${voicePath}, Text: ${voiceText}, Lang: ${voiceLang})`);
                    // You might want to send this to the backend for configuration
                }
            });
        }

        // Render in speaker voice dropdown
        if (speakerVoiceSelect) {
            speakerVoiceSelect.innerHTML = '<option value="">Select a voice</option>'; // Clear existing options
            voices.forEach(voice => {
                const option = document.createElement('option');
                option.value = voice.path;
                option.textContent = voice.name;
                option.dataset.voiceLang = voice.language; // Store language for later use
                speakerVoiceSelect.appendChild(option);
            });

            // Add event listener for speaker voice dropdown change
            speakerVoiceSelect.addEventListener('change', (event) => {
                const selectedPath = event.target.value;
                document.querySelectorAll('.voice-list li').forEach(item => {
                    if (item.dataset.voicePath === selectedPath) {
                        item.classList.add('selected');
                    } else {
                        item.classList.remove('selected');
                    }
                });
            });
        }
    }

    // Initial render of voices
    renderVoices().then(() => {
        // After rendering, if there's a pre-selected voice (e.g., from localStorage or default),
        // ensure it's highlighted and selected in the dropdown.
        // For now, we'll just ensure the dropdown change event is triggered if a value is set.
        if (speakerVoiceSelect.value) {
            speakerVoiceSelect.dispatchEvent(new Event('change'));
        }
    });

    // Initialize user info on page load
    const storedUsername = localStorage.getItem('userName');
    const storedEmail = localStorage.getItem('userEmail');
    if (storedUsername) {
        accountUsernameSpan.textContent = storedUsername;
        accountDisplayUsername.textContent = storedUsername;
    }
    if (storedEmail) {
        accountDisplayEmail.textContent = storedEmail;
    }

    // Function to update user icon based on theme
    const updateUserIcon = (theme) => {
        if (accountUserIcon) {
            // Corrected paths to be relative to the root /ui/images
            accountUserIcon.src = theme === 'dark' ? '/ui/images/white.png' : '/ui/images/black.png';
        }
    };

    // Initial update of user icon
    updateUserIcon(document.documentElement.getAttribute('data-theme') || 'light');

    // Listen for theme changes to update user icon
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.attributeName === 'data-theme') {
                updateUserIcon(document.documentElement.getAttribute('data-theme'));
            }
        });
    });
    observer.observe(document.documentElement, { attributes: true });


    // Account button and floating container interactions
    if (accountButton) {
        accountButton.addEventListener('click', (event) => {
            event.stopPropagation(); // Prevent document click from closing immediately
            accountFloatingContainer.classList.toggle('hidden');
            // Update account details from localStorage
            const storedUsername = localStorage.getItem('userName') || 'Guest';
            const storedEmail = localStorage.getItem('userEmail') || 'guest@example.com';
            accountDisplayUsername.textContent = storedUsername;
            accountDisplayEmail.textContent = storedEmail;

            const selectedVoiceOption = speakerVoiceSelect.options[speakerVoiceSelect.selectedIndex];
            const selectedVoiceName = selectedVoiceOption && selectedVoiceOption.value !== "" ? selectedVoiceOption.textContent : 'N/A';
            document.getElementById('accountDisplayVoice').textContent = selectedVoiceName;

            const selectedInputLanguageOption = inputLanguageSelect.options[inputLanguageSelect.selectedIndex];
            const selectedInputLanguageName = selectedInputLanguageOption ? selectedInputLanguageOption.textContent.split('(')[0].trim() : 'N/A'; // Extract only the language name
            document.getElementById('accountDisplayLanguage').textContent = selectedInputLanguageName;

            document.getElementById('accountDisplayTheme').textContent = document.documentElement.getAttribute('data-theme') === 'dark' ? 'Dark' : 'Light';
        });
    }

    // Close account floating container when clicking outside
    document.addEventListener('click', (event) => {
        if (accountFloatingContainer && !accountFloatingContainer.contains(event.target) && !accountButton.contains(event.target) && !accountFloatingContainer.classList.contains('hidden')) {
            accountFloatingContainer.classList.add('hidden');
        }
    });

    if (logoutButton) {
        logoutButton.addEventListener('click', () => {
            console.log('Logging out...');
            localStorage.removeItem('userToken');
            localStorage.removeItem('userName');
            localStorage.removeItem('userEmail');
            window.location.href = '/ui/auth/auth.html'; // Redirect to login page
        });
    }

    // Toggle visibility of speaker voice selection based on TTS model and re-initialize
    if (ttsModelSelect && mainControlsTopRow) {
        const updateVoiceSelectionVisibilityAndReinitialize = async () => {
            const currentTtsModelChoice = ttsModelSelect.value;
            if (currentTtsModelChoice === 'xtts') { // Show for XTTS
                voiceSelectionGroup.classList.remove('hidden');
                mainControlsTopRow.classList.add('expanded');
            } else {
                voiceSelectionGroup.classList.add('hidden');
                mainControlsTopRow.classList.remove('expanded');
            }

            // Trigger re-initialization when TTS model changes
            showNotification('TTS model changed. Re-initializing models...', 'info');
            const userToken = localStorage.getItem('userToken');
            if (!userToken) {
                showNotification('You must be logged in to re-initialize models.', 'error');
                return;
            }

            const sourceLang = inputLanguageSelect.value;
            const targetLang = document.getElementById('outputLanguageSelect').value;
            const ttsModelChoice = ttsModelSelect.value;
            const sttModelSize = "tiny"; // Default STT model size
            const vadEnabled = true; // Default VAD enabled state

            let speakerWavPath = null;
            let speakerText = null;
            let speakerLang = null;

            if (ttsModelChoice === 'xtts') { // Check for XTTS
                const selectedVoiceOption = speakerVoiceSelect.options[speakerVoiceSelect.selectedIndex];
                if (selectedVoiceOption && selectedVoiceOption.value) {
                    speakerWavPath = selectedVoiceOption.value;
                    speakerLang = selectedVoiceOption.dataset.voiceLang || 'en';
                } else {
                    showNotification(`Please select a voice for XTTS.`, 'warning');
                    // Do not return here, allow initialization with null speaker if XTTS is selected but no voice
                }
            }

            const queryParams = new URLSearchParams({
                source_lang: sourceLang,
                target_lang: targetLang,
                tts_model_choice: ttsModelChoice
            });

            try {
                const response = await fetch(`/initialize?${queryParams.toString()}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${userToken}`
                    },
                    body: JSON.stringify({
                        stt_model_size: sttModelSize,
                        vad_enabled_param: vadEnabled,
                        speaker_wav_path: speakerWavPath,
                        speaker_text: speakerText,
                        speaker_lang: speakerLang
                    })
                });

                if (response.ok) {
                    const result = await response.json();
                    showNotification('Models re-initialized successfully!', 'success');
                    console.log('Re-initialization result:', result);
                    // Immediate UI feedback for success
                    initBtn.disabled = true;
                    initBtn.classList.remove('btn-active');
                    startStopBtn.disabled = false;
                    startStopBtn.classList.add('btn-active');
                    statusLabel.textContent = 'Ready';
                    statusIndicator.classList.remove('off');
                    statusIndicator.classList.add('on');
                } else {
                    let errorData;
                    const contentType = response.headers.get('content-type');
                    if (contentType && contentType.includes('application/json')) {
                        errorData = await response.json();
                    } else { // If not JSON, it's plain text
                        errorData = await response.text();
                        showNotification(`Failed to re-initialize models: ${errorData}`, 'error'); // Display raw text directly
                    }
                    console.error('Re-initialization failed:', errorData);
                    // Immediate UI feedback for failure
                    initBtn.disabled = false;
                    initBtn.classList.add('btn-active');
                    startStopBtn.disabled = true;
                    startStopBtn.classList.remove('btn-active');
                    statusLabel.textContent = 'Error';
                    statusIndicator.classList.remove('on');
                    statusIndicator.classList.add('off');
                }
            } catch (error) {
                showNotification('Network error during model re-initialization.', 'error');
                console.error('Error during re-initialization:', error);
                // Immediate UI feedback for network error
                initBtn.disabled = false;
                initBtn.classList.add('btn-active');
                startStopBtn.disabled = true;
                startStopBtn.classList.remove('btn-active');
                statusLabel.textContent = 'Error';
                statusIndicator.classList.remove('on');
                statusIndicator.classList.add('off');
            }
        };

        ttsModelSelect.addEventListener('change', updateVoiceSelectionVisibilityAndReinitialize);
        // Set initial state
        updateVoiceSelectionVisibilityAndReinitialize();
    }

    if (recordVoiceBtn) {
        recordVoiceBtn.addEventListener('click', () => {
            voiceConfigModal.classList.add('open');
        });
    }

    if (closeModalBtn) {
        closeModalBtn.addEventListener('click', () => {
            voiceConfigModal.classList.remove('open');
        });
    }

    // Function to update button states based on checkbox
    const updateButtonStates = (isChecked) => {
        startRecordingModalBtn.disabled = !isChecked;
        uploadSpeakerVoiceModalBtn.disabled = !isChecked;
        // stopRecordingModalBtn remains hidden/disabled until recording starts
    };

    if (agreeToRecordCheckbox) {
        agreeToRecordCheckbox.addEventListener('change', () => {
            const isChecked = agreeToRecordCheckbox.checked;
            updateButtonStates(isChecked);
            // Stop microphone stream if checkbox is unchecked
            if (!isChecked && audioStream) {
                audioStream.getTracks().forEach(track => track.stop());
                audioStream = null;
                updateModalMicLevel(0); // Reset mic level display
                // Hide stop button and show start button if stream is stopped
                startRecordingModalBtn.classList.remove('hidden');
                stopRecordingModalBtn.classList.add('hidden');
            }
        });
        // Initial state check on load
        updateButtonStates(agreeToRecordCheckbox.checked);
    }

    // Recording functionality
    if (startRecordingModalBtn) {
        startRecordingModalBtn.addEventListener('click', () => {
            const modalRecordingStatus = document.getElementById('modalRecordingStatus');
            const recordingIndicator = document.getElementById('recordingIndicator'); // Get the indicator element
            console.log('Start Recording button clicked.');

            // Request microphone access and start recording
            navigator.mediaDevices.getUserMedia({ audio: true })
                .then(stream => {
                    if (!stream || !stream.active) {
                        throw new Error("Failed to get an active MediaStream from microphone.");
                    }
                    audioStream = stream;
                    // Initialize MediaRecorder without a specific MIME type, allowing the browser to choose a default.
                    // This matches mic_test.html for broader browser compatibility.
                    mediaRecorder = new MediaRecorder(audioStream);
                    console.log('MediaRecorder initialized with default MIME type:', mediaRecorder.mimeType);

                    console.log('Attempting to start recording...');
                    audioChunks = [];
                    mediaRecorder.ondataavailable = event => {
                        audioChunks.push(event.data);
                    };
                    mediaRecorder.onstop = () => {
                        console.log('MediaRecorder stopped. Creating audio blob.');
                        // Explicitly create a WAV blob, mirroring mic_test.html for backend compatibility.
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        currentRecordedBlob = audioBlob; // Store the blob
                        console.log('Recorded audio blob:', currentRecordedBlob);
                        // Trigger voice naming modal
                        voiceNamingModal.classList.add('open');
                        voiceConfigModal.classList.remove('open'); // Close config modal
                        voiceNameInput.value = ''; // Clear previous name
                        document.getElementById('voiceNameError').textContent = ''; // Clear error
                        // Reset recording UI state
                        startRecordingModalBtn.classList.remove('hidden');
                        stopRecordingModalBtn.classList.add('hidden');
                        startRecordingModalBtn.disabled = !agreeToRecordCheckbox.checked; // Re-enable start button based on checkbox
                        stopRecordingModalBtn.disabled = true;
                        modalRecordingStatus.textContent = 'Ready to record.';
                        modalRecordingStatus.style.color = 'var(--text-color)';
                        recordingIndicator.classList.remove('active'); // Deactivate indicator
                        // Ensure audio stream is stopped after recording
                        if (audioStream) {
                            audioStream.getTracks().forEach(track => track.stop());
                            audioStream = null;
                        }
                        updateModalMicLevel(0); // Reset mic level display
                    };
                    mediaRecorder.start();
                    console.log('MediaRecorder started. State:', mediaRecorder.state);
                    startRecordingModalBtn.classList.add('hidden');
                    stopRecordingModalBtn.classList.remove('hidden');
                    stopRecordingModalBtn.disabled = false; // Enable stop button
                    modalRecordingStatus.textContent = 'Recording...';
                    modalRecordingStatus.style.color = 'var(--danger-color)';
                    recordingIndicator.classList.add('active'); // Activate indicator

                    // Dynamic mic level visualization for modal
                    modalAudioContext = new (window.AudioContext || window.webkitAudioContext)();
                    modalMicrophone = modalAudioContext.createMediaStreamSource(audioStream);
                    modalAnalyser = modalAudioContext.createAnalyser();
                    modalAnalyser.fftSize = 256;
                    modalMicrophone.connect(modalAnalyser);

                    const bufferLength = modalAnalyser.frequencyBinCount;
                    const dataArray = new Uint8Array(bufferLength);

                    const updateModalVisualizer = () => {
                        modalAnalyser.getByteFrequencyData(dataArray);
                        let sum = 0;
                        for (let i = 0; i < bufferLength; i++) {
                            sum += dataArray[i];
                        }
                        const average = sum / bufferLength;
                        const level = Math.min(10, Math.floor(average / 25.5)); // Scale 0-255 to 0-10
                        updateModalMicLevel(level);
                        animationFrameId = requestAnimationFrame(updateModalVisualizer);
                    };
                    animationFrameId = requestAnimationFrame(updateModalVisualizer);

                })
                .catch(err => {
                    console.error('Error accessing microphone:', err);
                    showNotification('Microphone access denied or not available. Please enable it in your browser settings.', 'error');
                    agreeToRecordCheckbox.checked = false;
                    // Ensure audioStream is null if mic access fails
                    if (audioStream) {
                        audioStream.getTracks().forEach(track => track.stop());
                        audioStream = null;
                    }
                    updateButtonStates(false); // Disable buttons on error
                    updateModalMicLevel(0); // Reset mic level display
                    // Also reset button text and indicator if starting failed
                    startRecordingModalBtn.classList.remove('hidden');
                    stopRecordingModalBtn.classList.add('hidden');
                    startRecordingModalBtn.disabled = !agreeToRecordCheckbox.checked;
                    stopRecordingModalBtn.disabled = true;
                    modalRecordingStatus.textContent = 'Ready to record.';
                    modalRecordingStatus.style.color = 'var(--text-color)';
                    recordingIndicator.classList.remove('active');
                });
        });
    }

    if (stopRecordingModalBtn) {
        stopRecordingModalBtn.addEventListener('click', () => {
            console.log('Stop Recording button clicked.');
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                console.log('Attempting to stop recording...');
                mediaRecorder.stop();
                console.log('MediaRecorder stopped. State:', mediaRecorder.state);
                // The onstop event handler will reset the UI state and stop the audio stream
            } else {
                showNotification('Stop Recording clicked, but no active recording found.', 'warning');
                console.warn('Stop Recording clicked, but mediaRecorder is not in recording state. Resetting UI.');
                startRecordingModalBtn.classList.remove('hidden');
                stopRecordingModalBtn.classList.add('hidden');
                startRecordingModalBtn.disabled = !agreeToRecordCheckbox.checked;
                stopRecordingModalBtn.disabled = true;
                document.getElementById('modalRecordingStatus').textContent = 'Ready to record.';
                document.getElementById('modalRecordingStatus').style.color = 'var(--text-color)';
                document.getElementById('recordingIndicator').classList.remove('active');
                if (audioStream) {
                    audioStream.getTracks().forEach(track => track.stop());
                    audioStream = null;
                }
                updateModalMicLevel(0);
            }
        });
    }

    if (closeModalBtn) {
        closeModalBtn.addEventListener('click', () => {
            // Stop recording if active when modal is closed
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
            // Stop microphone stream if active
            if (audioStream) {
                audioStream.getTracks().forEach(track => track.stop());
                audioStream = null;
            }
            // Stop modal audio context and animation
            if (modalAudioContext) {
                modalAudioContext.close();
                modalAudioContext = null;
            }
            if (animationFrameId) {
                cancelAnimationFrame(animationFrameId);
                animationFrameId = null;
            }
            voiceConfigModal.classList.remove('open');
            agreeToRecordCheckbox.checked = false; // Reset checkbox state
            startRecordingModalBtn.disabled = true;
            stopRecordingModalBtn.disabled = true;
            startRecordingModalBtn.classList.remove('hidden');
            stopRecordingModalBtn.classList.add('hidden');
            uploadSpeakerVoiceModalBtn.disabled = true;
            updateModalMicLevel(0); // Reset mic level display
        });
    }

    // Cancel button in voice config modal
    const cancelModalBtn = document.getElementById('cancelModalBtn');
    if (cancelModalBtn) {
        cancelModalBtn.addEventListener('click', () => {
            // Stop recording if active when modal is cancelled
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
            // Stop microphone stream if active
            if (audioStream) {
                audioStream.getTracks().forEach(track => track.stop());
                audioStream = null;
            }
            voiceConfigModal.classList.remove('open');
            agreeToRecordCheckbox.checked = false; // Reset checkbox state
            startRecordingModalBtn.disabled = true;
            stopRecordingModalBtn.disabled = true;
            startRecordingModalBtn.classList.remove('hidden');
            stopRecordingModalBtn.classList.add('hidden');
            uploadSpeakerVoiceModalBtn.disabled = true;
            updateModalMicLevel(0); // Reset mic level display
        });
    }

    // Upload voice functionality
    if (uploadSpeakerVoiceModalBtn && speakerVoiceUploadModal && currentSpeakerVoiceModal) {
        speakerVoiceUploadModal.addEventListener('change', (event) => {
            const file = event.target.files[0];
            if (file) {
                currentSpeakerVoiceModal.textContent = `Selected: ${file.name}`;
                // Placeholder for actual upload logic
                console.log('File selected for upload:', file.name);
            } else {
                currentSpeakerVoiceModal.textContent = 'No voice selected';
            }
        });

        uploadSpeakerVoiceModalBtn.addEventListener('click', async () => {
            const userToken = localStorage.getItem('userToken');
            if (!userToken) {
                alert('You must be logged in to upload a voice.');
                return;
            }

            if (!speakerVoiceUploadModal.files.length) {
                speakerVoiceUploadModal.click();
            } else {
                const file = speakerVoiceUploadModal.files[0];
                const speakerLang = modalInputLanguageSelect.value; // Get selected language for the voice

                const formData = new FormData();
                formData.append('file', file);
                formData.append('speaker_lang', speakerLang);

                try {
                    const response = await fetch('/api/voices/upload', {
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${userToken}`
                        },
                        body: formData
                    });

                    const data = await response.json();
                    if (response.ok) {
                        console.log('Voice uploaded successfully:', data.voice);
                        voiceConfigModal.classList.remove('open');
                        renderVoices(); // Re-render sidebar voices
                        alert('Voice uploaded successfully!');
                    } else {
                        console.error('Failed to upload voice:', data.message);
                        alert(`Failed to upload voice: ${data.message}`);
                    }
                } catch (error) {
                    console.error('Error during voice upload:', error);
                    alert('Network error or server unreachable during voice upload.');
                }
            }
        });
    }

    // Translate reading statement based on selected language
    if (modalInputLanguageSelect && readingText) {
        const originalStatement = '"Reading this statement, I agree to provide my voice for cloning purposes. My voice will be used to synthesize translated speech within this application."';

        async function translateStatement(text, targetLang) {
            // Placeholder for actual backend MT module integration
            // In a real scenario, this would call a backend API that performs the translation
            console.log(`Simulating translation of "${text}" to ${targetLang}`);
            const simulatedTranslations = {
                "es": "Al leer esta declaración, acepto proporcionar mi voz para fines de clonación. Mi voz se utilizará para sintetizar el habla traducida dentro de esta aplicación.",
                "fr": "En lisant cette déclaration, j'accepte de fournir ma voix à des fins de clonage. Ma voix sera utilizada para sintetizar la parole traducida al seno de esta aplicación.",
                "de": "Mit dem Lesen dieser Erklärung stimme ich zu, meine Stimme für Klonierungszwecke zur Verfügung zu stellen. Meine Stimme wird verwendet, um übersetzte Sprache innerhalb dieser Anwendung zu synthetisieren.",
                "it": "Leggendo questa dichiarazione, accetto di fornire la mia voce per scopi di clonazione. La mia voce verrà utilizzata per sintetizzare il parlato tradotto all'interno di questa applicazione.",
                "pt": "Ao ler esta declaração, concordo em fornecer minha voz para fins de clonagem. Minha voz será usada para sintetizar a fala traduzida dentro deste aplicativo.",
                "pl": "Czytając to oświadczenie, zgadzam się udostępnić swój głos do celów klonowania. Mój głos zostanie użyty do syntezy przetłumaczonej mowy w tej aplikacji.",
                "tr": "Bu ifadeyi okuyarak, sesimi klonlama amacıyla sağlamayı kabul ediyorum. Sesim, bu uygulama içinde çevrilmiş konuşmayı sentezlemek için kullanılacaktır.",
                "ru": "Читая это заявление, я соглашаюсь предоставить свой голос для целей клоonирования. Мой голос будет использоваться для синтеза переведенной речи в этом приложении.",
                "nl": "Door deze verklaring te lezen, stem ik ermee in mijn stem te verstrekken voor kloningsdoeleinden. Mijn stem zal worden gebruikt om vertaalde spraak binnen deze applicatie te synthetiseren.",
                "cs": "Přečtením tohoto prohlášení souhlasím s poskytnutím svého hlasu pro účely klonování. Můj hlas bude použit k syntéze přeložené řeči v rámci této aplikace.",
                "ar": "بقراءة هذا البيان، أوافق على تقديم صوتي لأغراض الاستنساخ. سيتم استخدام صوتي لتوليف الكلام المترجم داخل هذا التطبيق.",
                "zh-cn": "通过阅读此声明，我同意提供我的声音用于克隆目的。我的声音将用于在此应用程序中合成翻译后的语音。",
                "ja": "この声明を読むことにより、私はクローン作成目的で自分の声を提供することに同意します。私の声は、このアプリケーション内で翻訳された音声を合成するために使用されます。",
                "hu": "Ezen nyilatkozat elolvasásával hozzájárulok ahhoz, hogy hangomat klónozási célokra felhasználják. Hangomat a lefordított beszéd szintetizálására fogják használni ebben az alkalmazásban.",
                "ko": "이 진술서를 읽음으로써, 저는 복제 목적으로 제 목소리를 제공하는 데 동의합니다. 제 목소리는 이 애플리케이션 내에서 번역된 음성을 합성하는 데 사용될 것입니다.",
                "hi": "इस कथन को पढ़कर, मैं क्लोनिंग उद्देश्यों के लिए अपनी आवाज़ प्रदान करने के लिए सहमत हूँ। मेरी आवाज़ का उपयोग इस एप्लिकेशन के भीतर अनुवादित भाषण को संश्लेषित करने के लिए किया जाएगा।",
                "sk": "Prečítaním tohto vyhlásenia súhlasím s poskytnutím svojho hlasu na účely klonovania. Môj hlas bude použitý na syntézu preloženej reči v rámci tejto aplikácie."
            };

            // Simulate a network delay
            await new Promise(resolve => setTimeout(resolve, 500));

            return simulatedTranslations[targetLang] || `[Translation to ${targetLang} not available] ${text}`;
        }

        modalInputLanguageSelect.addEventListener('change', async (event) => {
            const selectedLang = event.target.value;
            if (selectedLang === 'en') { // If English is selected, revert to original
                readingText.textContent = originalStatement;
            } else {
                const translatedText = await translateStatement(originalStatement, selectedLang);
                readingText.textContent = translatedText;
            }
        });
        // Set initial text
        readingText.textContent = originalStatement;
    }

    // Voice Naming Modal interactions
    if (saveVoiceNameBtn) {
        saveVoiceNameBtn.addEventListener('click', async () => {
            const newVoiceName = voiceNameInput.value.trim();
            if (!newVoiceName) {
                document.getElementById('voiceNameError').textContent = 'Voice name cannot be empty.';
                return;
            }
            document.getElementById('voiceNameError').textContent = ''; // Clear error

            const userToken = localStorage.getItem('userToken');
            if (!userToken) {
                showNotification('You must be logged in to save or rename a voice.', 'error');
                return;
            }

            if (voiceToEdit) { // Editing an existing voice
                try {
                    const response = await fetch('/api/voices/rename', {
                        method: 'PUT', // Corrected method to PUT
                        headers: {
                            'Content-Type': 'application/json',
                            'Authorization': `Bearer ${userToken}`
                        },
                        body: JSON.stringify({
                            old_name: voiceToEdit.name, // Backend expects voice name as old_name
                            new_name: newVoiceName // Backend expects voice name as new_name
                        })
                    });

                    const data = await response.json();
                    if (response.ok) {
                        console.log('Voice renamed successfully:', data.message);
                        showNotification('Voice renamed successfully!', 'success'); // Use custom notification
                        await renderVoices(); // Re-render sidebar voices and wait for it to complete
                        voiceNamingModal.classList.remove('open');

                        // Automatically select the renamed voice
                        const renamedVoicePath = voiceToEdit.path.replace(voiceToEdit.path.split('/').pop(), `${newVoiceName}.wav`);
                        const renamedVoiceLi = document.querySelector(`li[data-voice-path="${renamedVoicePath}"]`);
                        if (renamedVoiceLi) {
                            document.querySelectorAll('.voice-list li').forEach(item => item.classList.remove('selected'));
                            renamedVoiceLi.classList.add('selected');
                            speakerVoiceSelect.value = renamedVoicePath;
                            speakerVoiceSelect.dispatchEvent(new Event('change'));
                        }

                        voiceToEdit = null; // Clear voice being edited
                    } else {
                        const errorMessage = data.detail || data.message || 'Unknown error';
                        console.error('Failed to rename voice:', errorMessage);
                        showNotification(`Failed to rename voice: ${errorMessage}`, 'error'); // Use custom notification
                    }
                } catch (error) {
                    console.error('Error during voice rename:', error);
                    showNotification('Network error or server unreachable during voice rename.', 'error'); // Use custom notification
                }
            } else if (currentRecordedBlob) { // Saving a new recorded voice
                const speakerLang = modalInputLanguageSelect.value; // Get selected language for the voice

                const formData = new FormData();
                // Always send as .wav since we are creating a WAV blob now
                formData.append('file', currentRecordedBlob, `${newVoiceName}.wav`);
                formData.append('voice_name', newVoiceName); // Backend expects voice_name as a form field
                formData.append('speaker_lang', speakerLang);

                try {
                    const response = await fetch('/api/voices/upload', { // Corrected endpoint
                        method: 'POST',
                        headers: {
                            'Authorization': `Bearer ${userToken}`
                        },
                        body: formData
                    });

                    const data = await response.json();
                    if (response.ok) {
                        console.log('Voice uploaded successfully:', data.voice);
                        showNotification('Voice uploaded successfully!', 'success'); // Use custom notification
                        renderVoices(); // Re-render sidebar voices
                        voiceNamingModal.classList.remove('open');
                        currentRecordedBlob = null; // Clear recorded blob
                        // Reset recording UI state
                        startRecordingModalBtn.classList.remove('hidden'); // Show start button
                        stopRecordingModalBtn.classList.add('hidden'); // Hide stop button
                        startRecordingModalBtn.disabled = !agreeToRecordCheckbox.checked; // Re-enable start button based on checkbox
                        stopRecordingModalBtn.disabled = true;
                        document.getElementById('modalRecordingStatus').textContent = 'Ready to record.';
                        document.getElementById('modalRecordingStatus').style.color = 'var(--text-color)';
                    } else {
                        const errorMessage = data.detail || data.message || 'Unknown error';
                        console.error('Failed to upload voice:', errorMessage);
                        showNotification(`Failed to upload voice: ${errorMessage}`, 'error'); // Use custom notification
                    }
                } catch (error) {
                    console.error('Error during voice upload:', error);
                    showNotification('Network error or server unreachable during voice upload.', 'error'); // Use custom notification
                }
            }
        });
    }

    if (cancelVoiceNameBtn) {
        cancelVoiceNameBtn.addEventListener('click', () => {
            voiceNamingModal.classList.remove('open');
        });
    }

    if (closeNamingModalBtn) {
        closeNamingModalBtn.addEventListener('click', () => {
            voiceNamingModal.classList.remove('open');
        });
    }

    // Delete Confirmation Modal interactions
    if (confirmDeleteBtn) {
        confirmDeleteBtn.addEventListener('click', async () => {
            if (!voiceToEdit) {
                showNotification('No voice selected for deletion.', 'warning');
                return;
            }

            const userToken = localStorage.getItem('userToken');
            if (!userToken) {
                showNotification('You must be logged in to delete a voice.', 'error');
                return;
            }

            try {
                const response = await fetch('/api/voices/delete', { // Corrected endpoint
                    method: 'DELETE', // Corrected method to DELETE
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${userToken}`
                    },
                    body: JSON.stringify({ filename: voiceToEdit.filename }) // Backend expects filename
                });

                const data = await response.json();
                if (response.ok) {
                    console.log('Voice deleted successfully:', data.message);
                    showNotification('Voice deleted successfully!', 'success');
                    await renderVoices(); // Re-render sidebar voices and wait for it to complete
                    deleteConfirmationModal.classList.remove('open');
                    voiceToEdit = null; // Clear voice being deleted

                    // After deletion, if the deleted voice was the currently selected one,
                    // reset the speaker voice selection and account display.
                    if (speakerVoiceSelect.value === voiceToEdit.path) {
                        speakerVoiceSelect.value = ""; // Select the "Select a voice" option
                        speakerVoiceSelect.dispatchEvent(new Event('change'));
                    }
                } else {
                    console.error('Failed to delete voice:', data.message);
                    showNotification(`Failed to delete voice: ${data.message}`, 'error');
                }
            } catch (error) {
                console.error('Error during voice deletion:', error);
                showNotification('Network error or server unreachable during voice deletion.', 'error');
            }
        });
    }

    if (cancelDeleteBtn) {
        cancelDeleteBtn.addEventListener('click', () => {
            deleteConfirmationModal.classList.remove('open');
        });
    }

    if (closeDeleteModalBtn) {
        closeDeleteModalBtn.addEventListener('click', () => {
            deleteConfirmationModal.classList.remove('open');
        });
    }

    // Example for input level bar (simplified)
    if (inputLevelSegments) {
        for (let i = 0; i < 10; i++) { // 10 segments
            const segment = document.createElement('div');
            segment.classList.add('segment');
            inputLevelSegments.appendChild(segment);
        }
        // In a real scenario, update segments based on actual mic input
        // setInterval(() => {
        //     const level = Math.floor(Math.random() * 11); // 0-10
        //     Array.from(inputLevelSegments.children).forEach((segment, index) => {
        //         if (index < level) {
        //             segment.classList.add('active');
        //         } else {
        //             segment.classList.remove('active');
        //         }
        //     });
        // }, 200);
    }

    if (modalInputLevelSegments) {
        for (let i = 0; i < 10; i++) { // 10 segments
            const segment = document.createElement('div');
            segment.classList.add('segment');
            modalInputLevelSegments.appendChild(segment);
        }
    }

    // Chart.js for Latency Timeline
    // Chart.js for Latency Timeline
    function initializeLatencyChart() {
        const latencyTimelineChartCanvas = document.getElementById('latencyTimelineChart');
        if (latencyTimelineChartCanvas) {
            const ctx = latencyTimelineChartCanvas.getContext('2d');
            latencyChart = new Chart(ctx, { // Assign to the global latencyChart
                type: 'line',
                data: {
                    labels: Array.from({ length: 300 }, (_, i) => `${i}s`), // 0s to 5 minutes (300 seconds)
                    datasets: [
                        {
                            label: 'Input',
                            data: [],
                            borderColor: 'rgba(75, 192, 192, 1)',
                            backgroundColor: 'rgba(75, 192, 192, 0.2)',
                            fill: true,
                            stepped: true,
                        },
                        {
                            label: 'Translation Output',
                            data: [],
                            borderColor: 'rgba(153, 102, 255, 1)',
                            backgroundColor: 'rgba(153, 102, 255, 0.2)',
                            fill: true,
                            stepped: true,
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'linear', // Change to linear scale
                            min: 0,
                            max: 300, // Show 5 minutes (300 seconds) of data
                            title: {
                                display: true,
                                text: 'Time (seconds)'
                            },
                            ticks: {
                                stepSize: 30 // Show ticks every 30 seconds
                            }
                        },
                        y: {
                            beginAtZero: true,
                            max: 2, // 0 for idle, 1 for input, 2 for translation
                            ticks: {
                                callback: function (value, index, ticks) {
                                    if (value === 0) return 'Idle';
                                    if (value === 1) return 'Input';
                                    if (value === 2) return 'Translation';
                                    return '';
                                }
                            },
                            title: {
                                display: true,
                                text: 'Activity'
                            }
                        }
                    },
                    animation: false // Disable animation for real-time feel
                }
            });

            // Function to update chart data (to be integrated with actual pipeline events)
            window.updateLatencyChart = function (timeInSeconds, inputActive, translationActive) {
                if (!latencyChart) return;

                const inputDataset = latencyChart.data.datasets[0];
                const translationDataset = latencyChart.data.datasets[1];

                // Add new data points
                inputDataset.data.push({ x: timeInSeconds, y: inputActive ? 1 : 0 });
                translationDataset.data.push({ x: timeInSeconds, y: translationActive ? 2 : 0 });

                // Keep only the last 300 seconds (5 minutes) of data
                const maxDataPoints = 300;
                if (inputDataset.data.length > maxDataPoints) {
                    inputDataset.data.shift();
                    translationDataset.data.shift();
                }

                // Update x-axis min/max to follow the latest data point, keeping a 5-minute window
                const latestTime = timeInSeconds;
                latencyChart.options.scales.x.min = Math.max(0, latestTime - maxDataPoints);
                latencyChart.options.scales.x.max = latestTime;

                latencyChart.update();
            };

            // Function to reset chart data
            window.resetLatencyChart = function () {
                if (!latencyChart) return;

                latencyChart.data.labels = [];
                latencyChart.data.datasets[0].data = [];
                latencyChart.data.datasets[1].data = [];

                // Reset scales
                latencyChart.options.scales.x.min = 0;
                latencyChart.options.scales.x.max = 300;

                latencyChart.update();
            };
        }
    }

    // Initialize the chart
    initializeLatencyChart();
});
