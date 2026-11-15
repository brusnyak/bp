document.addEventListener('DOMContentLoaded', () => {
    const ttsModelSelect = document.getElementById('ttsModelSelect');
    const inputLanguageSelect = document.getElementById('inputLanguageSelect');
    const voiceSelectionGroup = document.getElementById('voice-selection-group');
    const speakerVoiceSelect = document.getElementById('speakerVoiceSelect');
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
    const voiceNamingModal = document.getElementById('voiceNamingModal');
    const closeNamingModalBtn = document.getElementById('closeNamingModalBtn');
    const saveVoiceNameBtn = document.getElementById('saveVoiceNameBtn');
    const cancelVoiceNameBtn = document.getElementById('cancelVoiceNameBtn');
    const inputLevelSegments = document.getElementById('inputLevelSegments');
    const modalInputLevelSegments = document.getElementById('modalInputLevelSegments');
    const voiceNameInput = document.getElementById('voiceNameInput'); // Added for voice naming modal


    let referenceRecordTimer;
    let mediaRecorder;
    let audioChunks = [];
    let audioStream; // To store the microphone stream
    let currentRecordedBlob = null; // To store the recorded audio blob temporarily
    let notificationTimeout; // For custom notifications
    let websocket; // Declare websocket globally
    let latencyChart = null; // Declare latencyChart globally within the DOMContentLoaded scope

    // Global variables for main audio pipeline
    let mainAudioContext = null;
    let mainAnalyser = null;
    let mainMicrophone = null;
    let mainGainNode = null; // Declare mainGainNode globally
    let mainAudioWorkletNode = null; // Replaced ScriptProcessorNode
    let monitorSink = null;

    // --- Input level smoothing state (matching BP xtts) ---
    let emaLevel = 0.0; // exponential moving average for smoothing
    const EMA_ALPHA = 0.3; // Smoothing factor
    const NUM_LEVEL_SEGMENTS = 10; // Number of mic level segments
    let mainAudioStream = null; // To store the microphone stream for the main pipeline
    let chartIdleInterval = null; // Interval for updating chart during idle state
    let sessionStartTime = 0; // Track session start time for relative chart timing

    // --- Throttle state for debug logs (avoid spamming console) ---
    let lastLogAt = 0;
    const LOG_THROTTLE_MS = 1000; // log at most once per second


    // Activity Log Helper Function with colored event types
    function logActivity(message, type = 'info') {
        const activityLog = document.getElementById('activityLog');
        if (!activityLog) return;
        
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry log-${type}`; // CSS classes: log-info, log-success, log-warning, log-error
        logEntry.textContent = `[${timestamp}] ${message}`;
        activityLog.appendChild(logEntry);
        activityLog.scrollTop = activityLog.scrollHeight; // Auto-scroll to bottom
    }

    // Audio output device management
    const audioOutputSelect = document.getElementById('audioOutputSelect');
    const virtualMicHint = document.getElementById('virtualMicHint');
    let selectedAudioOutputDeviceId = null;

    // Enumerate and populate audio output devices
    async function enumerateAudioOutputDevices() {
        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const inputSelect = document.getElementById('audioInputSelect');
            const previousInput = inputSelect.value;
            inputSelect.replaceChildren(new Option('System default input', ''));
            for (const input of devices.filter(device => device.kind === 'audioinput')) {
                inputSelect.add(new Option(input.label || 'Audio input', input.deviceId));
            }
            if ([...inputSelect.options].some(option => option.value === previousInput)) inputSelect.value = previousInput;
            const audioOutputs = devices.filter(device => device.kind === 'audiooutput');

            // Rebuild the list without silently moving playback to a virtual cable.
            // Rehearsals should stay audible in headphones unless the user explicitly
            // chooses their meeting cable. A prior explicit selection is restored.
            const preferredOutput = localStorage.getItem('preferredAudioOutput');
            audioOutputSelect.innerHTML = '<option value="">Speakers / headphones (default)</option>';
            selectedAudioOutputDeviceId = null;

            let virtualDeviceFound = false;
            let virtualDeviceName = null;

            audioOutputs.forEach(device => {
                const option = document.createElement('option');
                option.value = device.deviceId;
                option.textContent = device.label || `Audio Output ${audioOutputs.indexOf(device) + 1}`;
                audioOutputSelect.appendChild(option);

                // Auto-detect virtual device (BlackHole or VB-Cable)
                const label = device.label.toLowerCase();
                if (label.includes('blackhole') || label.includes('vb-cable') || label.includes('vb cable') || label.includes('vb-audio')) {
                    virtualDeviceFound = true;
                    virtualDeviceName = device.label;
                }
                if (preferredOutput && device.deviceId === preferredOutput) {
                    option.selected = true;
                    selectedAudioOutputDeviceId = device.deviceId;
                }
            });

            // Update hint text based on detection
            if (virtualDeviceFound) {
                virtualMicHint.textContent = `Virtual output available: ${virtualDeviceName}. Choose it only when sending translated audio into a meeting.`;
                virtualMicHint.style.color = 'var(--log-success-color)';
                // Availability is not a successful routing check.
            } else if (audioOutputs.length && audioOutputs.every(d => !d.label)) {
                virtualMicHint.textContent = 'Device names are hidden until the browser has microphone permission. Run the microphone test to detect a virtual cable.';
                virtualMicHint.style.color = 'var(--secondary-color)';
            } else {
                virtualMicHint.textContent = 'Playback stays on speakers/headphones. A virtual cable (BlackHole, VB-Cable) is needed only to send audio into a meeting; select the cable as your meeting microphone.';
                virtualMicHint.style.color = 'var(--secondary-color)'; // informational, not a warning (orange failed contrast)
                console.warn('Frontend: No virtual audio device detected. Translation will play through default output.');
            }

            // A disconnected saved device must not become an invisible routing choice.
            if (preferredOutput && !selectedAudioOutputDeviceId) localStorage.removeItem('preferredAudioOutput');

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


    function showNotification(message, type = 'info') {
        if (type === 'error' || type === 'warning') setRouteStatus(message);
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
        inputLevelSegments.setAttribute('aria-valuenow', String(Math.round(normalizedLevel * 100)));
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

    const routeStatus = document.getElementById('routeStatus');
    let reconnectTimer = null;
    let modelsReady = false;
    let captureGeneration = 0;
    let soundCheckStream = null;
    let soundCheckRecorder = null;

    let starting = false;
    // Playback timing goes to the server session log (BP_LOG_SESSIONS=1) so lag can be split into chunking, recognition and playback queueing.
    const playback = new SessionPlayback({ adaptive: true, makeAudio: url => {
        const audio = new Audio(url);
        audio.addEventListener('ended', () => { if (isStreaming) setRouteStatus('Listening'); });
        return audio;
    }, onEvent: e => { if (isStreaming && e.event === 'tts_play_start') setRouteStatus('Speaking');  try { if (websocket && websocket.readyState === 1) websocket.send(JSON.stringify({ type: 'client_event', t_ms: Math.round(performance.timeOrigin + performance.now()), ...e })); } catch (_) {} }, onError: error => {
        stopRealtimeAudioStreaming();

        updateStartAvailability();
        showNotification(error.message, 'error');
    }});
    const rateSelect = document.getElementById('playbackRateSelect');
    if (rateSelect) {
        try { rateSelect.value = localStorage.getItem('bp.playbackRate') || 'auto'; } catch { /* storage unavailable */ }
        const applyRate = () => {
            playback.adaptive = rateSelect.value === 'auto';
            playback.playbackRate = Number(rateSelect.value) || 1;
            try { localStorage.setItem('bp.playbackRate', rateSelect.value); } catch { /* storage unavailable */ }
        };
        rateSelect.addEventListener('change', applyRate);
        applyRate();
    }
    const setRouteStatus = text => { if (routeStatus) routeStatus.textContent = text; };
    const deviceFingerprint = () => `${document.getElementById('audioInputSelect').value}|${selectedAudioOutputDeviceId || ''}`;
    let pendingPreparation = null;
    const finishPreparation = ready => {
        if (!pendingPreparation) return;
        clearTimeout(pendingPreparation.timer);
        pendingPreparation.resolve(ready);
        pendingPreparation = null;
    };
    const publishLive = () => {
        window.__live = { isStreaming, starting, modelsReady };
        document.dispatchEvent(new Event('livestate'));
    };
    const updateStartAvailability = () => {
        startStopBtn.disabled = !isStreaming && !starting && (!websocket || websocket.readyState !== WebSocket.OPEN);
        startStopBtn.textContent = isStreaming || starting ? 'Stop' : 'Start';
        const label = audioOutputSelect.selectedOptions[0]?.textContent || '';
        document.getElementById('feedbackWarning').hidden = !(isStreaming && !/headphone|headset|blackhole|vb[- ]?(cable|audio)/i.test(label));
        publishLive();
    };
    for (const id of ['audioInputSelect', 'audioOutputSelect']) document.getElementById(id).addEventListener('change', () => {
        stopRealtimeAudioStreaming();
        document.getElementById('soundCheckStatus').textContent = '';
    });
    document.getElementById('soundCheckBtn').addEventListener('click', async () => {
        stopRealtimeAudioStreaming();
        const generation = captureGeneration;
        const button = document.getElementById('soundCheckBtn');
        const status = document.getElementById('soundCheckStatus');
        const fingerprint = deviceFingerprint();

        button.disabled = true;
        let checkedStream = null;
        let recorder = null;
        try {
            const input = document.getElementById('audioInputSelect').value;
            const stream = await navigator.mediaDevices.getUserMedia({ audio: input ? { deviceId: { exact: input }, echoCancellation: true } : true });
            if (generation !== captureGeneration) { stream.getTracks().forEach(track => track.stop()); return; }
            checkedStream = stream;
            soundCheckStream = stream;
            recorder = new MediaRecorder(stream);
            soundCheckRecorder = recorder;
            const chunks = [];
            recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
            const recorded = new Promise((resolve, reject) => { recorder.onstop = resolve; recorder.onerror = event => reject(event.error || new Error('Recording failed')); });
            recorder.start();
            status.textContent = 'Say a short phrase. Recording locally for two seconds…';
            await new Promise(resolve => setTimeout(resolve, 2000));
            if (recorder.state !== 'inactive') recorder.stop();
            await recorded;
            stream.getTracks().forEach(track => track.stop());
            soundCheckStream = null;
            if (generation !== captureGeneration || fingerprint !== deviceFingerprint()) return;
            await enumerateAudioOutputDevices();
            if (generation !== captureGeneration || fingerprint !== deviceFingerprint()) return;

            updateStartAvailability();
            playback.start();
            playback.enqueue(new Blob(chunks, { type: recorder.mimeType }), selectedAudioOutputDeviceId || '');
            status.textContent = 'Check that you hear your phrase on the intended output, use headphones if needed. This checks devices, not translation quality.';
        } catch (error) { status.textContent = `Sound check failed: ${error.message}`; }
        finally {
            if (recorder && recorder.state !== 'inactive') recorder.stop();
            if (checkedStream) checkedStream.getTracks().forEach(track => track.stop());
            if (soundCheckStream === checkedStream) soundCheckStream = null;
            button.disabled = false;
            updateStartAvailability();
        }
    });
    navigator.mediaDevices?.addEventListener('devicechange', () => {
        stopRealtimeAudioStreaming();
        enumerateAudioOutputDevices();
        setRouteStatus('Audio devices changed. Check input/output and run the sound check again.');
    });
    // Each browser session gets a distinct, short-lived anonymous token.
    let guestRetried = false;
    async function startGuestSession() {
        try {
            const r = await fetch('/api/guest', { method: 'POST' });
            if (!r.ok) return false;
            const d = await r.json();
            localStorage.setItem('userToken', d.token);
            return true;
        } catch (e) { return false; }
    }
    // WebSocket setup
    function setupWebSocket() {
        clearTimeout(reconnectTimer);
        const token = localStorage.getItem('userToken');
        if (!token) {
            setRouteStatus('Connecting…');
            startGuestSession().then(ok => {
                if (ok) setupWebSocket();
                else setRouteStatus('Local session unavailable. Check the translation service configuration and reload.');
            });
            return;
        }
        const scheme = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const WS_URL = `${scheme}//${window.location.host}/ws?token=${encodeURIComponent(token)}`;
        websocket = new WebSocket(WS_URL);

        websocket.onopen = (event) => {
            modelsReady = false;
            updateStartAvailability();
            setRouteStatus('Ready');
            renderVoices();
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
                    if (!pendingPreparation) return;
                    if (message.route_status) logActivity(JSON.stringify(message.route_status), 'info');
                    modelsReady = Boolean(message.fully_loaded);
                    if (modelsReady) {
                        finishPreparation(true);
                        if (!starting) setRouteStatus('Ready');
                        updateStartAvailability();
                        logActivity('Route prepared', 'success');
                    } else if (starting) {
                        setRouteStatus('Preparing…');
                    }
                } else if (message.type === 'error') {
                    modelsReady = false;
                    stopRealtimeAudioStreaming();
                    updateStartAvailability();
                    showNotification(message.message || 'Route preparation failed. Check installed assets and prepare again.', 'error');
                } else if (message.type === 'caption_partial') {
                    document.getElementById('transcriptionOutput').textContent = message.text;
                    updatePip(message.text, undefined);
                } else if (message.type === 'transcription_result') {
                    document.getElementById('transcriptionOutput').textContent = message.transcribed;
                    updatePip(message.transcribed, undefined);
                    
                    if (isStreaming) setRouteStatus('Translating');

                    if (latencyChart) latencyChart.update();
                } else if (message.type === 'translation_result') {
                    document.getElementById('translationOutput').textContent = message.translated;
                    updatePip(undefined, message.translated);
                    
                    if (latencyChart) latencyChart.update();
                } else if (message.type === 'final_metrics') {
                    console.log('Frontend: Final Metrics:', message.metrics);
                    
                    // Update text metrics
                    const sttLatencyEl = document.getElementById('sttLatency');
                    const mtLatencyEl = document.getElementById('mtLatency');
                    const ttsLatencyEl = document.getElementById('ttsLatency');
                    const totalLatencyEl = document.getElementById('totalLatency');

                    if (sttLatencyEl) sttLatencyEl.textContent = (message.metrics.stt_time || 0).toFixed(2) + 's';
                    if (mtLatencyEl) mtLatencyEl.textContent = (message.metrics.mt_time || 0).toFixed(2) + 's';
                    if (ttsLatencyEl) ttsLatencyEl.textContent = (message.metrics.tts_time || 0).toFixed(2) + 's';
                    if (totalLatencyEl) totalLatencyEl.textContent = (message.metrics.total_latency || 0).toFixed(2) + 's';

                    // Update activity log
                    const activityLog = document.getElementById('activityLog');
                    if (activityLog) {
                        const timestamp = new Date().toLocaleTimeString();
                        const logEntry = document.createElement('div');
                        logEntry.className = 'log-entry';
                        logEntry.textContent = `[${timestamp}] Processed speech - STT: ${(message.metrics.stt_time || 0).toFixed(2)}s, MT: ${(message.metrics.mt_time || 0).toFixed(2)}s, TTS: ${(message.metrics.tts_time || 0).toFixed(2)}s`;
                        activityLog.appendChild(logEntry);
                        activityLog.scrollTop = activityLog.scrollHeight; // Auto-scroll to bottom
                    }

                    // Update settings status
                    const settingsStatusText = document.getElementById('settingsStatusText');
                    if (settingsStatusText) {
                        settingsStatusText.textContent = `Last processed: ${new Date().toLocaleTimeString()} | Total: ${(message.metrics.total_latency || 0).toFixed(2)}s`;
                    }

                } else if (message.type === 'lag_trace') {
                    window.updateLatencyChart(message);
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
            } else if (event.data instanceof Blob || event.data instanceof ArrayBuffer) {
                if (isStreaming) playback.enqueue(new Blob([event.data], { type: 'audio/wav' }), selectedAudioOutputDeviceId || '');
            }
        };

        websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            showNotification('WebSocket error. Please check console for details.', 'error');
        };

        websocket.onclose = (event) => {
            modelsReady = false;
            stopRealtimeAudioStreaming();
            updateStartAvailability();
            if ([1008, 4003, 4401, 4403].includes(event.code)) {
                const wasGuest = localStorage.getItem('userGuest') === '1';
                localStorage.removeItem('userToken');
                if (wasGuest && !guestRetried) { guestRetried = true; setupWebSocket(); return; }
                setRouteStatus('Local session expired or rejected. Reload to reconnect.');
                return;
            }
            showNotification('Disconnected from translation service.', 'warning');
            // Attempt to reconnect after a delay
            setRouteStatus('Connection lost. Capture stopped. Reconnecting…');
            reconnectTimer = setTimeout(setupWebSocket, 5000);
        };
    }

    // Call setupWebSocket on page load
    setupWebSocket();
    window.addEventListener('pagehide', () => {
        stopRealtimeAudioStreaming(); clearTimeout(reconnectTimer);
        if (websocket) { websocket.onclose = null; websocket.close(); }
    });

    const startStopBtn = document.getElementById('startStopBtn');

    let isStreaming = false; // Track if real-time streaming is active
    let modalAudioContext = null; // For modal mic level visualization
    let modalAnalyser = null;
    let modalMicrophone = null;
    let animationFrameId = null; // To cancel animation frame for modal mic level

    // Set initial button states
    updateStartAvailability();
    startStopBtn.disabled = true;
    startStopBtn.classList.remove('btn-active');

    // Function to start real-time audio streaming
    async function startRealtimeAudioStreaming() {
        console.log('Frontend: Attempting to start real-time audio streaming...'); // Added log
        const generation = captureGeneration;
        try {
            console.log('Frontend: Calling navigator.mediaDevices.getUserMedia...'); // Added log before getUserMedia


            if (!modelsReady || !websocket || websocket.readyState !== WebSocket.OPEN) throw new Error('Route unavailable. Press Start to prepare it again.');
            const deviceId = document.getElementById('audioInputSelect').value;
            // Echo cancellation, noise suppression and automatic gain are OFF by default (the page warns about feedback with speakers);
            // ?mic=processed switches them back on. The choice is written to the session log as a client event.
            const rawMic = new URLSearchParams(location.search).get('mic') !== 'processed';   // raw is the default since the 2026-11-08 A/B (English, same engine: 2.7% error raw against 6.7% processed)
            const audioConstraints = { audio: {
                ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
                echoCancellation: !rawMic, noiseSuppression: !rawMic, autoGainControl: !rawMic
            }};
            try { if (websocket && websocket.readyState === 1) websocket.send(JSON.stringify({ type: 'client_event', event: rawMic ? 'mic_raw' : 'mic_processed', t_ms: Math.round(performance.timeOrigin + performance.now()) })); } catch (_) {}
            const capturedStream = await navigator.mediaDevices.getUserMedia(audioConstraints);
            if (generation !== captureGeneration) { capturedStream.getTracks().forEach(track => track.stop()); return; }
            mainAudioStream = capturedStream;
            mainAudioStream.getAudioTracks().forEach(track => track.addEventListener('ended', () => {
                if (isStreaming) { stopRealtimeAudioStreaming(); showNotification('Audio input disconnected. Select a device and prepare again.', 'error'); }
            }));
            console.log('Frontend: MediaStream obtained. Active:', mainAudioStream.active, 'Tracks:', mainAudioStream.getAudioTracks());
            // Mic permission unlocks real device labels/ids: re-enumerate outputs so the
            // playback dropdown carries usable deviceIds (pre-permission they are empty).
            try { await enumerateAudioOutputDevices(); } catch (e) { console.warn('Frontend: output re-enumeration failed:', e); }
            if (generation !== captureGeneration) return;
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
            await mainAudioContext.audioWorklet.addModule('/ui/live-speech/audio-processor.js');
            if (generation !== captureGeneration) return;
            if (mainAudioContext.sampleRate !== 16000) throw new Error('This browser did not provide 16 kHz capture. Use a supported browser.');
            mainMicrophone = mainAudioContext.createMediaStreamSource(mainAudioStream);
            mainAnalyser = mainAudioContext.createAnalyser();
            mainGainNode = mainAudioContext.createGain(); // Assign to global mainGainNode
            mainGainNode.gain.value = 1.0; // Increased gain for better mic pickup (was 2.0 = 6dB, now 4.0 = 12dB boost)
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

                if (!isStreaming) return;

                // The AudioWorkletNode already outputs Float32Array, which is what the backend expects.
                // Sending the raw Float32Array buffer directly.
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.send(inputBuffer.buffer); // Send raw Float32Array buffer
                }
            };

            mainMicrophone.connect(mainGainNode); // Connect microphone to GainNode
            mainGainNode.connect(mainAnalyser); // Connect GainNode to Analyser
            mainAnalyser.connect(mainAudioWorkletNode); // Connect analyser to AudioWorkletNode
            // Keep the worklet pulled without audible microphone monitoring.
            monitorSink = mainAudioContext.createGain();
            monitorSink.gain.value = 0;
            mainAudioWorkletNode.connect(monitorSink);
            monitorSink.connect(mainAudioContext.destination);

            // Set session start time for chart
            if (typeof window.updateLatencyChart === 'function') {
                sessionStartTime = performance.now() / 1000;
            }

            // Send "start" command to backend to begin transcription
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(JSON.stringify({ type: "start" }));
                console.log('Frontend: Sent "start" command to backend');
            }

            isStreaming = true; publishLive();
            playback.start();
            startStopBtn.textContent = 'Stop';
            setRouteStatus('Listening');
            updateStartAvailability();
            showNotification('Real-time streaming started.', 'success');
            logActivity('Session started - real-time streaming active', 'success');
        } catch (err) {
            if (generation !== captureGeneration) return;
            console.error('Error starting real-time audio streaming:', err);
            showNotification(err.message || 'Microphone access failed. Check permissions and prepare again.', 'error');
            stopRealtimeAudioStreaming(); // Ensure cleanup on error
        }
    }

    // Function to stop real-time audio streaming
    function stopRealtimeAudioStreaming() {
        const wasStreaming = isStreaming || starting;
        captureGeneration++;
        finishPreparation(false);
        starting = false;
        isStreaming = false; publishLive();
        playback.stop();
        if (soundCheckRecorder && soundCheckRecorder.state !== 'inactive') soundCheckRecorder.stop();
        if (soundCheckStream) { soundCheckStream.getTracks().forEach(track => track.stop()); soundCheckStream = null; }
        if (monitorSink) { monitorSink.disconnect(); monitorSink = null; }
        if (websocket && websocket.readyState === WebSocket.OPEN) websocket.send(JSON.stringify({ type: 'stop' }));
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

        isStreaming = false; publishLive();
        startStopBtn.textContent = 'Start';
        setRouteStatus('Ready');
        updateMainMicLevel(0); // Reset mic level display
        updateStartAvailability();

        // Reset chart if it exists
        if (typeof window.resetLatencyChart === 'function') {
            window.resetLatencyChart();
        }

        // Only report "stopped" when we were actually streaming — not on page load or idle teardowns.
        if (wasStreaming) {
            showNotification('Real-time streaming stopped.', 'info');
            logActivity('Session ended - streaming stopped', 'info');
        }
    }

    // Start prepares the selected route, then opens capture. Stop is terminal at every await.
    startStopBtn.addEventListener('click', async () => {
        if (isStreaming || starting) { stopRealtimeAudioStreaming(); return; }
        stopRealtimeAudioStreaming();
        const generation = captureGeneration;
        starting = true;
        updateStartAvailability();
        try {
            if (!modelsReady) {
                setRouteStatus('Preparing…');
                const prepared = new Promise(resolve => {
                    pendingPreparation = { resolve, timer: setTimeout(() => {
                        finishPreparation(false);
                        if (generation === captureGeneration) setRouteStatus('Preparation timed out. Press Start to retry.');
                    }, 120000) };
                });
                const sent = await sendConfigUpdate();
                if (generation !== captureGeneration) return;
                if (!sent) finishPreparation(false);
                if (!await prepared || generation !== captureGeneration) return;
            }
            if (generation === captureGeneration) await startRealtimeAudioStreaming();
        } catch (error) {
            if (generation === captureGeneration) { stopRealtimeAudioStreaming(); showNotification(error.message, 'error'); }
        } finally {
            if (generation === captureGeneration) { starting = false; updateStartAvailability(); }
        }
    });

    // PiP floating subtitles (spec 002 US1): hidden canvas-fed video popped out via
    // requestPictureInPicture. Mirrors the subtitle panels (transcript + translation) so
    // captions follow across tabs/windows/apps with zero installs. Needs a user gesture;
    // unsupported browsers get a notice instead of silence.
    let pipVideo = null, pipCanvas = null, pipCtx = null;
    const pipLines = { transcript: '', translation: '' };
    const PIP_W = 640, PIP_H = 280; // wider, more room for 2-3 lines each

    function wrapPipText(ctx, text, maxWidth, maxLines) {
        const words = String(text || '').split(/\s+/).filter(Boolean);
        const lines = [];
        let line = '';
        words.forEach((w) => {
            const test = line ? line + ' ' + w : w;
            if (ctx.measureText(test).width > maxWidth && line) {
                lines.push(line);
                line = w;
            } else {
                line = test;
            }
        });
        if (line) lines.push(line);
        return lines.slice(-maxLines);
    }

    function drawPip() {
        if (!pipCtx) return;
        const ctx = pipCtx;
        const w = pipCanvas.width, h = pipCanvas.height;

        // Dark glass background with subtle vignette
        const bg = ctx.createLinearGradient(0, 0, 0, h);
        bg.addColorStop(0, 'rgba(8, 16, 40, 0.95)');
        bg.addColorStop(1, 'rgba(4, 8, 20, 0.98)');
        ctx.fillStyle = bg;
        ctx.fillRect(0, 0, w, h);

        // Top accent bar (brand color)
        ctx.fillStyle = '#C41E3A';
        ctx.fillRect(0, 0, w, 4);

        // Subtle inner glow at top
        const glow = ctx.createLinearGradient(0, 4, 0, 24);
        glow.addColorStop(0, 'rgba(196, 30, 58, 0.15)');
        glow.addColorStop(1, 'rgba(196, 30, 58, 0)');
        ctx.fillStyle = glow;
        ctx.fillRect(0, 4, w, 20);

        // Title
        ctx.fillStyle = 'rgba(255,255,255,0.6)';
        ctx.font = '500 13px system-ui, -apple-system, sans-serif';
        ctx.fillText('Hlas · live subtitles', 20, 26);

        const hasText = (pipLines.transcript || pipLines.translation).trim();
        let y = 56;
        const lineH = 36;
        const maxW = w - 40;

        if (!hasText) {
            ctx.fillStyle = 'rgba(255,255,255,0.4)';
            ctx.font = '400 18px system-ui, -apple-system, sans-serif';
            ctx.fillText('…waiting for speech…', 20, y + lineH);
            return;
        }

        // Source transcript (white, slightly larger)
        ctx.fillStyle = '#ffffff';
        ctx.font = '400 22px system-ui, -apple-system, sans-serif';
        wrapPipText(ctx, pipLines.transcript, maxW, 3).forEach((l) => {
            // Subtle text shadow for readability
            ctx.shadowColor = 'rgba(0,0,0,0.6)';
            ctx.shadowBlur = 2;
            ctx.shadowOffsetX = 0;
            ctx.shadowOffsetY = 1;
            ctx.fillText(l, 20, y);
            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;
            y += lineH;
        });

        // Divider
        y += 6;
        ctx.strokeStyle = 'rgba(196, 30, 58, 0.5)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(20, y);
        ctx.lineTo(w - 20, y);
        ctx.stroke();
        y += 10;

        // Translation (green, slightly smaller)
        ctx.fillStyle = '#5eead4'; // softer green
        ctx.font = '400 20px system-ui, -apple-system, sans-serif';
        wrapPipText(ctx, pipLines.translation, maxW, 3).forEach((l) => {
            ctx.shadowColor = 'rgba(0,0,0,0.6)';
            ctx.shadowBlur = 2;
            ctx.shadowOffsetX = 0;
            ctx.shadowOffsetY = 1;
            ctx.fillText(l, 20, y);
            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;
            y += lineH;
        });
    }

    function updatePip(transcript, translation) {
        if (transcript !== undefined) pipLines.transcript = transcript;
        if (translation !== undefined) pipLines.translation = translation;
        if (pipVideo && document.pictureInPictureElement === pipVideo) drawPip();
    }

    const pipBtn = document.getElementById('pipBtn');
    if (pipBtn) {
        pipBtn.addEventListener('click', async () => {
            try {
                if (document.pictureInPictureElement) {
                    await document.exitPictureInPicture();
                    return;
                }
                if (!('pictureInPictureEnabled' in document) || !document.pictureInPictureEnabled) {
                    showNotification('Picture-in-Picture is not supported in this browser. The in-page subtitle panels still work.', 'error');
                    return;
                }
                if (!pipCanvas) {
                    pipCanvas = document.createElement('canvas');
                    pipCanvas.width = PIP_W;
                    pipCanvas.height = PIP_H;
                    pipCtx = pipCanvas.getContext('2d');
                    drawPip(); // paint BEFORE capture: Chrome only fires loadedmetadata once a frame exists
                    pipVideo = document.createElement('video');
                    pipVideo.muted = true;
                    pipVideo.playsInline = true;
                    pipVideo.srcObject = pipCanvas.captureStream();
                    // Canvas-fed streams carry no loaded metadata at creation: PiP
                    // requested before readyState>=1 throws InvalidStateError
                    // ("Metadata for the video element are not loaded yet").
                    await pipVideo.play();
                    if (pipVideo.readyState < 1) {
                        await new Promise((resolve) => {
                            const done = () => {
                                pipVideo.removeEventListener('loadedmetadata', done);
                                resolve();
                            };
                            pipVideo.addEventListener('loadedmetadata', done);
                            setTimeout(done, 2000); // never trap the button on odd browsers
                        });
                    }
                }
                await pipVideo.requestPictureInPicture();
                showNotification('Subtitles popped out — the window stays visible across apps.', 'success');
            } catch (err) {
                console.warn('Frontend: PiP failed:', err);
                showNotification('Could not pop out subtitles: ' + (err && err.message ? err.message : err), 'error');
            }
        });
    }

    // Function to send configuration update via WebSocket
    async function sendConfigUpdate() {
        if (!websocket || websocket.readyState !== WebSocket.OPEN) {
            console.warn('Frontend: WebSocket not ready. Cannot send config update.');
            showNotification('WebSocket not connected. Cannot update configuration.', 'error');
            return false;
        }

        const sourceLang = inputLanguageSelect.value;
        const targetLang = document.getElementById('outputLanguageSelect').value;
        const generation = captureGeneration;
        modelsReady = false;
        updateStartAvailability();
        try {
            const response = await fetch('/api/routes', { cache: 'no-store' });
            if (!response.ok) throw new Error('Cannot check installed route assets. Retry preparation.');
            const inventory = await response.json();
            if (generation !== captureGeneration) return false;
            const route = inventory.routes.find(route => route.source === sourceLang && route.target === targetLang);
            if (!route) throw new Error('Choose an explicitly declared input and target language pair. Automatic or undeclared routes need evaluation first.');
            if (!route.assets.resolved) throw new Error('Setup required before this route can run: ' + route.assets.errors.join('; '));
        } catch (error) {
            if (generation === captureGeneration) showNotification(error.message, 'error');
            return false;
        }
        const ttsModelChoice = ttsModelSelect.value;
        const vadEnabled = true; // Always enabled for now
        
        // Get VAD settings if available (or use defaults)
        const vadAggressiveness = 3; 
        const silenceRmsThreshold = 0.02;

        let speakerWavPath = null;
        let speakerText = null;
        let speakerLang = null;

        if (['xtts', 'omnivoice'].includes(ttsModelChoice)) {
            const selectedVoiceOption = speakerVoiceSelect.options[speakerVoiceSelect.selectedIndex];
            if (selectedVoiceOption && selectedVoiceOption.value) {
                speakerWavPath = selectedVoiceOption.value;
                speakerLang = modalInputLanguageSelect.value || sourceLang;
            }
        }

        const configMessage = {
            type: "config_update", // Correct message type for backend
            source_lang: sourceLang,
            target_lang: targetLang,
            tts_model_choice: ttsModelChoice,
            speaker_wav_path: speakerWavPath,
            speaker_text: speakerText,
            speaker_lang: speakerLang,
            vad_enabled: vadEnabled,
            vad_aggressiveness: vadAggressiveness,
            silence_rms_threshold: silenceRmsThreshold,
            strategy: (document.getElementById('strategySelect') || {}).value || undefined,
            ...((() => {
                const r = (document.getElementById('roomIdInput')?.value.trim() || new URLSearchParams(location.search).get('room') || '').trim();
                return r ? { room_id: r } : {};
            })())
        };

        if (generation !== captureGeneration || websocket.readyState !== WebSocket.OPEN) return false;
        websocket.send(JSON.stringify(configMessage));
        
        // Update UI to loading state
        updateStartAvailability();
        startStopBtn.classList.remove('btn-active');
        setRouteStatus('Preparing…');
        return true;
    }

    const invalidatePreparation = () => {
        stopRealtimeAudioStreaming(); modelsReady = false;
        updateStartAvailability();
        setRouteStatus('Ready');
    };
    // Selection never silently starts model loading.
    const strategySelectEl = document.getElementById('strategySelect');
    if (strategySelectEl) {
        strategySelectEl.addEventListener('change', invalidatePreparation);
        // Server decides the default per machine class (BP_DEFAULT_STRATEGY in presets/*.env).
        fetch('/api/session/config').then((r) => r.json()).then((c) => {
            if (c.default_strategy && [...strategySelectEl.options].some((o) => o.value === c.default_strategy)) strategySelectEl.value = c.default_strategy;
        }).catch(() => {});
    }
    if (inputLanguageSelect) {
        inputLanguageSelect.addEventListener('change', invalidatePreparation);
    }
    const outputLanguageSelect = document.getElementById('outputLanguageSelect');
    if (outputLanguageSelect) {
        outputLanguageSelect.addEventListener('change', invalidatePreparation);
    }

    // Keep the EN/SK badges beside the column titles in sync with the selects (they were static markup: "CS" showed for Slovak)
    const syncLanguageBadges = () => {
        [['inputLanguageSelect', 'inputLanguageBadge'], ['outputLanguageSelect', 'outputLanguageBadge']].forEach(([selId, badgeId]) => {
            const sel = document.getElementById(selId);
            const badge = document.getElementById(badgeId);
            if (sel && badge) badge.textContent = sel.value === 'auto' ? 'AUTO' : sel.value.toUpperCase();
        });
    };
    ['inputLanguageSelect', 'outputLanguageSelect'].forEach((id) => {
        const sel = document.getElementById(id);
        if (sel) sel.addEventListener('change', syncLanguageBadges);
    });
    syncLanguageBadges();

    // Direction label and swap: one click turns English -> Slovak into Slovak -> English (and back).
    const swapBtn = document.getElementById('swapLangBtn');
    const dirLabel = document.getElementById('directionLabel');
    const outSel = document.getElementById('outputLanguageSelect');
    const T = (x) => (window.i18n ? window.i18n.t(x) : x);
    const langName = (sel) => sel.value === 'auto' ? T('Auto-detect') : T(sel.options[sel.selectedIndex].text.split(' (')[0]);
    const syncDirection = () => { if (dirLabel && inputLanguageSelect && outSel) dirLabel.textContent = `${langName(inputLanguageSelect)} → ${langName(outSel)}`; };
    const hasOption = (sel, v) => [...sel.options].some((o) => o.value === v);
    if (swapBtn && inputLanguageSelect && outSel) {
        swapBtn.addEventListener('click', () => {
            const a = inputLanguageSelect.value, b = outSel.value;
            const newIn = b;
            const newOut = a === 'auto' ? (b === 'en' ? 'sk' : 'en') : a;
            if (!hasOption(inputLanguageSelect, newIn) || !hasOption(outSel, newOut)) return;
            inputLanguageSelect.value = newIn;
            outSel.value = newOut;
            inputLanguageSelect.dispatchEvent(new Event('change'));
            outSel.dispatchEvent(new Event('change'));
        });
    }
    [inputLanguageSelect, outSel].forEach((sel) => sel && sel.addEventListener('change', syncDirection));
    document.addEventListener('languagechange', syncDirection);
    syncDirection();

    ttsModelSelect.addEventListener('change', () => {
        voiceSelectionGroup.classList.remove('hidden');
        invalidatePreparation();
    });
    speakerVoiceSelect.addEventListener('change', invalidatePreparation);
    document.getElementById('deleteVoiceReferenceBtn').addEventListener('click', async () => {
        const name = speakerVoiceSelect.value;
        if (!name) return;
        try {
            const response = await fetch('/api/voice-reference?name=' + encodeURIComponent(name), {
                method: 'DELETE', headers: { Authorization: 'Bearer ' + localStorage.getItem('userToken') }
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.detail || 'Could not delete reference');
            await renderVoices();
            invalidatePreparation();
        } catch (error) { showNotification(error.message, 'error'); }
    });

    // Function to load voices from backend API
    async function loadVoices() {
        const userToken = localStorage.getItem('userToken');
        if (!userToken) {
            return [];
        }

        try {
            const response = await fetch('/api/voice-reference', {
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

    // Local reference files are shared by this installation.
    async function renderVoices(selected) {
        const voices = await loadVoices();
        const previous = selected || speakerVoiceSelect.value;
        speakerVoiceSelect.replaceChildren(new Option('Select a voice reference', ''));
        voices.forEach(voice => {
            const option = new Option(voice.name, voice.name);
            speakerVoiceSelect.add(option);
        });
        if ([...speakerVoiceSelect.options].some(option => option.value === previous)) speakerVoiceSelect.value = previous;
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

    // Recording-passage tabs: same per-language scripts the voices train on.
    // (Old 2-sentence consent text retired 2026-09: too short to clone from.)
    fetch('passages.json', { cache: 'no-store' }).then((r) => r.json()).then((passages) => {
        const box = document.getElementById('readingText');
        const tabs = Array.from(document.querySelectorAll('.passage-tab'));
        const langSelect = document.getElementById('modalInputLanguageSelect');
        // One language control: tabs and dropdown mirror each other (en/sk/cs only).
        const show = (lang) => {
            if (box && passages[lang]) box.textContent = passages[lang];
            if (langSelect && langSelect.querySelector(`option[value="${lang}"]`)) langSelect.value = lang;
            tabs.forEach((b) => b.classList.toggle('active', b.dataset.lang === lang));
        };
        tabs.forEach((b) => b.addEventListener('click', () => show(b.dataset.lang)));
        if (langSelect) langSelect.addEventListener('change', () => show(langSelect.value));
        show((langSelect && langSelect.value) || 'en');
    }).catch(() => {
        const box = document.getElementById('readingText');
        if (box) box.textContent = 'Could not load passages.json.';
    });

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
                        clearTimeout(referenceRecordTimer);
                        console.log('MediaRecorder stopped. Creating audio blob.');
                        // Keep the browser recorder format; changing a MIME label does not convert audio.
                        const audioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || audioChunks[0]?.type || 'audio/webm' });
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
                    referenceRecordTimer = setTimeout(() => { if (mediaRecorder.state === 'recording') mediaRecorder.stop(); }, 60000);
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
                showNotification('Voice upload unavailable. Reconnect the local session and retry.', 'error');
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
                    const response = await fetch('/api/voice-reference', {
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
                        await renderVoices(data.name);
                        invalidatePreparation();
                        alert('Voice uploaded successfully!');
                    } else {
                        console.error('Failed to upload voice:', data.message);
                        showNotification(response.status === 403 ? 'Voice upload is unavailable for this local session.' : `Failed to upload voice: ${data.detail || data.message || response.status}`, 'error');
                    }
                } catch (error) {
                    console.error('Error during voice upload:', error);
                    alert('Network error or server unreachable during voice upload.');
                }
            }
        });
    }

    // NOTE: a previous simulated consent-statement translator used to fight the
    // passage tabs over #readingText (hardcoded placeholder "translations", one of
    // them garbled). Removed 2026-09-29: passages.json is the single source.

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
                showNotification('Voice upload unavailable. Reconnect the local session and retry.', 'error');
                return;
            }

            if (currentRecordedBlob) { // Saving a new recorded voice
                const speakerLang = modalInputLanguageSelect.value; // Get selected language for the voice

                const formData = new FormData();
                const extension = /mp4|m4a/.test(currentRecordedBlob.type) ? 'm4a' : /mpeg/.test(currentRecordedBlob.type) ? 'mp3' : /wav/.test(currentRecordedBlob.type) ? 'wav' : 'webm';
                formData.append('file', currentRecordedBlob, `${newVoiceName}.${extension}`);
                formData.append('voice_name', newVoiceName); // Backend expects voice_name as a form field
                formData.append('speaker_lang', speakerLang);

                try {
                    const response = await fetch('/api/voice-reference', { // Corrected endpoint
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
                        await renderVoices(data.name);
                        invalidatePreparation();
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
                        const errorMessage = response.status === 403 ? 'Voice upload is unavailable for this local session.' : data.detail || data.message || 'Unknown error';
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

    // Server-send lag only. Browser output and physical-device queues are unmeasured.
    function initializeLatencyChart() {
        const canvas = document.getElementById('latencyTimelineChart');
        if (!canvas) return;
        const readout = document.getElementById('lagTraceSummary');
        const limit = 30;
        const traces = [];
        let fragmentNumber = 0;
        const parts = ['hold_ms', 'stt_ms', 'mt_ms', 'tts_first_ms'];
        const labels = ['Hold', 'STT', 'MT', 'TTS-first', 'Queue / other (unattributed)'];
        const tokenNames = ['--primary-color', '--log-success-color', '--secondary-color', '--deep-berry', '--border-color'];
        const finite = value => typeof value === 'number' && Number.isFinite(value);
        const fmt = value => finite(value) ? `${Math.round(value)} ms` : 'unmeasured';
        const percentile = (sorted, p) => {
            const index = (sorted.length - 1) * p;
            const lo = Math.floor(index), hi = Math.ceil(index);
            return sorted[lo] + (sorted[hi] - sorted[lo]) * (index - lo);
        };
        const overlay = {
            id: 'fragmentLagLabels',
            afterDatasetsDraw(chart) {
                const {ctx, chartArea, scales: {x, y}} = chart;
                ctx.save();
                ctx.fillStyle = chart.options.color;
                ctx.strokeStyle = chart.options.color;
                ctx.setLineDash([5, 4]);
                ctx.beginPath();
                ctx.moveTo(chartArea.left, y.getPixelForValue(2000));
                ctx.lineTo(chartArea.right, y.getPixelForValue(2000));
                ctx.stroke();
                ctx.setLineDash([]);
                ctx.textAlign = 'right';
                ctx.fillText('Goal 2000 ms', chartArea.right, y.getPixelForValue(2000) - 5);
                ctx.textAlign = 'center';
                traces.forEach((trace, i) => {
                    const total = trace.total_ms;
                    ctx.save();
                    // Vertical labels remain separate even with 30 fragments on a phone.
                    ctx.translate(x.getPixelForValue(i), Math.max(chartArea.top + 4,
                        y.getPixelForValue(finite(total) && total >= 0 ? total : 0) - 5));
                    ctx.rotate(-Math.PI / 2);
                    ctx.textAlign = 'left';
                    ctx.fillText(finite(total) ? `${Math.round(total)} ms` : 'n/a', 0, 0);
                    ctx.restore();
                });
                ctx.restore();
            }
        };
        latencyChart = new Chart(canvas.getContext('2d'), {
            type: 'bar',
            data: {labels: [], datasets: labels.map(label => ({label, data: [], stack: 'lag', borderWidth: 1}))},
            plugins: [overlay],
            options: {
                responsive: true, maintainAspectRatio: false, animation: false,
                layout: {padding: {top: 55}},
                scales: {
                    x: {stacked: true, title: {display: true, text: 'Fragment (last 30)'}},
                    y: {stacked: true, beginAtZero: true, suggestedMax: 2500,
                        title: {display: true, text: 'Speech end → first server audio send (ms)'}}
                },
                plugins: {
                    legend: {position: 'bottom', labels: {boxWidth: 10}},
                    tooltip: {
                        callbacks: {
                            title: items => traces[items[0].dataIndex].text,
                            label: item => `${item.dataset.label}: ${fmt(item.raw)}`,
                            afterBody: items => {
                                const trace = traces[items[0].dataIndex];
                                return [
                                    ...parts.map((key, i) => `${labels[i]} measured: ${fmt(trace[key])}`),
                                    `Total: ${fmt(trace.total_ms)}`,
                                    trace.overlap ? 'Parts overlap/exceed total; total shown as unattributed.' : 'Other includes unmeasured work; browser playback is excluded.'
                                ];
                            }
                        }
                    }
                }
            }
        });
        function applyTheme() {
            const style = getComputedStyle(canvas);
            const ink = style.getPropertyValue('--text-color').trim() || style.color;
            const border = style.getPropertyValue('--border-color').trim();
            latencyChart.options.color = ink;
            latencyChart.options.plugins.legend.labels.color = ink;
            for (const axis of Object.values(latencyChart.options.scales)) {
                axis.ticks.color = ink;
                axis.title.color = ink;
                axis.grid.color = border;
            }
            latencyChart.data.datasets.forEach((dataset, i) => {
                dataset.backgroundColor = style.getPropertyValue(tokenNames[i]).trim() || ink;
                dataset.borderColor = ink;
            });
            latencyChart.update('none');
        }
        function summarize() {
            const valid = traces.filter(t => finite(t.total_ms) && t.total_ms >= 0);
            if (!valid.length) {
                readout.textContent = 'Waiting for measured fragment lag. Window: last 30 fragments. Browser playback excluded.';
                return;
            }
            const sorted = valid.map(t => t.total_ms).sort((a, b) => a - b);
            const timed = valid.filter(t => finite(t.speech_end));
            let drift = null;
            if (timed.length > 1) {
                const meanX = timed.reduce((sum, t) => sum + t.speech_end, 0) / timed.length;
                const meanY = timed.reduce((sum, t) => sum + t.total_ms, 0) / timed.length;
                const variance = timed.reduce((sum, t) => sum + (t.speech_end - meanX) ** 2, 0);
                if (variance > 0) drift = timed.reduce((sum, t) => sum +
                    (t.speech_end - meanX) * (t.total_ms - meanY), 0) / variance * 60000;
            }
            readout.textContent = `Last ${traces.length}/30 fragments · ${valid.length} measured · Median ${fmt(percentile(sorted, .5))} · p90 ${fmt(percentile(sorted, .9))} · Drift ${finite(drift) ? `${Math.round(drift)} ms/min of speech` : 'unmeasured'} · Server send; browser playback excluded.`;
        }
        window.updateLatencyChart = function (trace) {
            // Legacy five-timestamp calls cannot supply fragment lag; never invent it.
            if (!trace || trace.type !== 'lag_trace') return;
            const measured = parts.map(key => finite(trace[key]) && trace[key] >= 0 ? trace[key] : null);
            const known = measured.reduce((sum, value) => sum + (value ?? 0), 0);
            const validTotal = finite(trace.total_ms) && trace.total_ms >= 0;
            const overlap = validTotal && known > trace.total_ms;
            traces.push({...trace, number: ++fragmentNumber, overlap});
            if (traces.length > limit) traces.shift();
            latencyChart.data.labels = traces.map(t => `#${t.number}`);
            latencyChart.data.datasets.forEach((dataset, i) => {
                dataset.data = traces.map(t => {
                    if (!finite(t.total_ms) || t.total_ms < 0) return null;
                    if (i === 4) return t.overlap ? t.total_ms : t.total_ms - parts.reduce((sum, key) =>
                        sum + (finite(t[key]) && t[key] >= 0 ? t[key] : 0), 0);
                    return t.overlap ? null : (finite(t[parts[i]]) && t[parts[i]] >= 0 ? t[parts[i]] : null);
                });
            });
            summarize();
            latencyChart.update('none');
        };
        window.resetLatencyChart = function () {
            traces.length = 0;
            fragmentNumber = 0;
            latencyChart.data.labels = [];
            latencyChart.data.datasets.forEach(dataset => { dataset.data = []; });
            summarize();
            latencyChart.update('none');
        };
        new MutationObserver(applyTheme).observe(document.documentElement,
            {attributes: true, attributeFilter: ['data-theme', 'class']});
        applyTheme();
        summarize();
    }

    // Initialize the chart
    initializeLatencyChart();

    // Room mode broadcast UI wireup
    const roomIdInput = document.getElementById('roomIdInput');
    const copyRoomLinkBtn = document.getElementById('copyRoomLinkBtn');
    const roomStatusInfo = document.getElementById('roomStatusInfo');
    const roomListenerCount = document.getElementById('roomListenerCount');
    const roomPreviewLink = document.getElementById('roomPreviewLink');

    if (roomIdInput) {
        const urlRoom = new URLSearchParams(location.search).get('room');
        if (urlRoom) roomIdInput.value = urlRoom;

        const updateRoomUI = () => {
            const rid = roomIdInput.value.trim();
            if (rid) {
                if (roomStatusInfo) roomStatusInfo.style.display = 'block';
                if (roomPreviewLink) { roomPreviewLink.hidden = false; roomPreviewLink.href = `/ui/listen/listen.html?room=${encodeURIComponent(rid)}`; }
            } else {
                if (roomStatusInfo) roomStatusInfo.style.display = 'none';
                if (roomPreviewLink) roomPreviewLink.hidden = true;
            }
        };

        roomIdInput.addEventListener('input', () => {
            updateRoomUI();
            invalidatePreparation();
        });
        updateRoomUI();

        // Start room: the server creates the room for this session (a typed code is refused: listeners cannot create rooms, speakers cannot join others').
        let roomListenUrl = '';
        const startRoomBtn = document.getElementById('startRoomBtn');
        if (startRoomBtn) {
            startRoomBtn.addEventListener('click', async () => {
                const auth = { Authorization: `Bearer ${localStorage.getItem('userToken') || ''}` };
                try {
                    const r = await fetch('/api/room', { method: 'POST', headers: auth });
                    if (!r.ok) throw new Error(r.status === 401 ? 'Local session unavailable. Reload and retry.' : `Server said ${r.status}`);
                    const made = await r.json();
                    roomIdInput.value = made.room_id; roomListenUrl = made.listen_url;
                    updateRoomUI(); invalidatePreparation();
                    showNotification('Room started. Listeners join by link.', 'success');
                } catch (e) {
                    showNotification(`Could not start a room: ${e.message}`, 'error');
                }
            });
        }

        if (copyRoomLinkBtn) {
            copyRoomLinkBtn.addEventListener('click', async () => {
                const rid = roomIdInput.value.trim();
                if (!rid) {
                    showNotification('Press Start room first.', 'warning');
                    return;
                }
                const url = roomListenUrl || `${location.origin}/ui/listen/listen.html?room=${encodeURIComponent(rid)}`;
                try {
                    await navigator.clipboard.writeText(url);
                    const oldText = copyRoomLinkBtn.textContent;
                    copyRoomLinkBtn.textContent = 'Copied!';
                    showNotification(`Copied audience link: ${url}`, 'success');
                    setTimeout(() => { copyRoomLinkBtn.textContent = oldText; }, 2000);
                } catch (e) {
                    showNotification(`Share link: ${url}`, 'info');
                }
            });
        }

        // Periodic listener count update when room is configured
        const pollRoomStatus = async () => {
            const rid = roomIdInput.value.trim();
            if (!rid) return;
            const token = localStorage.getItem('userToken');
            try {
                const res = await fetch(`/api/room/${encodeURIComponent(rid)}`, {
                    headers: token ? { 'Authorization': `Bearer ${token}` } : {}
                });
                if (res.ok) {
                    const data = await res.json();
                    if (roomListenerCount) {
                        const langsStr = Object.entries(data.languages || {})
                            .map(([l, count]) => `${l.toUpperCase()}: ${count}`).join(', ');
                        roomListenerCount.textContent = `${data.listeners} listener${data.listeners === 1 ? '' : 's'}${langsStr ? ' (' + langsStr + ')' : ''}`;
                    }
                }
            } catch (err) {
                // quiet background poll
            }
        };
        setInterval(pollRoomStatus, 5000);
    }

    // Log initial page load
    logActivity('Application loaded', 'success');
});


// Charts need a real size: nudge them when the collapsed diagnostics panel is opened.
(() => { const d = document.getElementById('diagnostics'); if (d) d.addEventListener('toggle', () => window.dispatchEvent(new Event('resize'))); })();
