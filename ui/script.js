document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Element References ---
    const inputLanguageSelect = document.getElementById('inputLanguageSelect');
    const outputLanguageSelect = document.getElementById('outputLanguageSelect');
    const ttsModelSelect = document.getElementById('ttsModelSelect');
    const initBtn = document.getElementById('initBtn');

    // Main content elements (initially hidden)
    const contentMain = document.querySelector('.content-main');
    const stateListening = document.getElementById('stateListening');
    const stateTranslating = document.getElementById('stateTranslating');
    const stateSpeaking = document.getElementById('stateSpeaking');
    const inputLevelSegmentsContainer = document.getElementById('inputLevelSegments');
    const NUM_LEVEL_SEGMENTS = 20; // Increased for more reactivity
    let levelSegments = [];
    const inputLanguageBadge = document.getElementById('inputLanguageBadge');
    const outputLanguageBadge = document.getElementById('outputLanguageBadge');
    const transcriptionBox = document.getElementById('transcriptionBox');
    const translationBox = document.getElementById('translationBox');
    const sttTime = document.getElementById('sttTime');
    const mtTime = document.getElementById('mtTime');
    const ttsTime = document.getElementById('ttsTime');
    const totalTime = document.getElementById('totalTime');
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    const statusIndicator = document.getElementById('statusIndicator');
    const statusLabel = document.getElementById('statusLabel');
    const settingsStatusText = document.getElementById('settingsStatusText');
    const activityLog = document.getElementById('activityLog');

    // Modal elements
    const voiceTrainingModal = document.getElementById('voiceTrainingModal');
    const modalInputLanguageSelect = document.getElementById('modalInputLanguageSelect'); // New
    const modalPhoneticPromptText = document.getElementById('modalPhoneticPromptText');
    const modalNotNowBtn = document.getElementById('modalNotNowBtn');
    const modalRecordBtn = document.getElementById('modalRecordBtn'); // New button
    const modalRecordingConfirmation = document.getElementById('modalRecordingConfirmation'); // New element for confirmation
    const modalChooseFileBtn = document.getElementById('modalChooseFileBtn'); // New button
    const modalReferenceAudioUpload = document.getElementById('modalReferenceAudioUpload');
    const modalReferenceAudioStatus = document.getElementById('modalReferenceAudioStatus');
    const modalLoadingIndicator = document.getElementById('modalLoadingIndicator'); // New
    const modalLoadingMessage = document.getElementById('modalLoadingMessage'); // New
    const modalMicLevelContainer = document.getElementById('modalMicLevelContainer'); // New
    const modalInputLevelSegmentsContainer = document.getElementById('modalInputLevelSegments'); // New
    let modalLevelSegments = []; // New for modal mic level

    // --- Global State ---
    let websocket = null;
    let audioContext = null;
    let mediaStream = null;
    let audioProcessor = null;
    let isInitialized = false;
    let isRecording = false;
    let isModalRecording = false; // New state for modal recording
    let currentPhoneticPrompt = null; // Store the generated phonetic prompt for the modal
    let referenceAudioBase64 = null; // Store base64 for reference audio from modal
    let referenceAudioMimeType = null; // Store mime type for reference audio from modal
    let modalAudioContext = null; // New AudioContext for modal recording
    let modalAudioProcessor = null; // New AudioProcessor for modal recording
    let modalMediaStream = null; // New MediaStream for modal recording
    let modalEmaLevel = 0.0; // EMA for modal mic level
    let modalCurrentWordIndex = 0; // Track current word for highlighting in modal

    // Build API / WS URLs using current page protocol and host (safer on dev vs prod)
    const proto = window.location.protocol === 'https:' ? 'https' : 'http';
    const wsProto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const API_BASE_URL = `${proto}://${window.location.hostname}${window.location.port ? ':' + window.location.port : ''}`;
    const WS_URL = `${wsProto}://${window.location.hostname}${window.location.port ? ':' + window.location.port : ''}/ws`;
    const API_URL = `${API_BASE_URL}`;

    // Initialize main UI level segments
    for (let i = 0; i < NUM_LEVEL_SEGMENTS; i++) {
        const segment = document.createElement('div');
        segment.classList.add('level-segment');
        inputLevelSegmentsContainer.appendChild(segment);
        levelSegments.push(segment);
    }

    // Initialize modal UI level segments
    for (let i = 0; i < NUM_LEVEL_SEGMENTS; i++) {
        const segment = document.createElement('div');
        segment.classList.add('level-segment');
        modalInputLevelSegmentsContainer.appendChild(segment);
        modalLevelSegments.push(segment);
    }

    const SAMPLE_RATE = 16000;
    const BUFFER_SIZE = 4096;

    // --- Input level smoothing state ----
    let emaLevel = 0.0; // exponential moving average for smoothing
    const EMA_ALPHA = 0.5; // Increased for more responsiveness

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
            element.classList.remove('idle');
            element.classList.add('active');
        } else {
            element.classList.remove('active');
            element.classList.add('idle');
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
                segment.classList.add('active');

                // style classes for medium/high ranges
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

    function setModalInputLevel(level) {
        const scaled = Math.sqrt(Math.min(1, Math.max(0, level)));
        modalEmaLevel = modalEmaLevel * (1 - EMA_ALPHA) + scaled * EMA_ALPHA;

        const normalizedLevel = Math.min(1, Math.max(0, modalEmaLevel));
        const activeSegments = Math.ceil(normalizedLevel * NUM_LEVEL_SEGMENTS);

        modalLevelSegments.forEach((segment, index) => {
            if (index < activeSegments) {
                segment.classList.add('active');
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

    function resetUI() {
        // Reset main UI elements
        transcriptionBox.innerHTML = '<p class="placeholder">Waiting for speech...</p>';
        translationBox.innerHTML = '<p class="placeholder">Translation will appear here...</p>';
        sttTime.textContent = '0.0s';
        mtTime.textContent = '0.0s';
        ttsTime.textContent = '0.0s';
        totalTime.textContent = '0.0s';
        emaLevel = 0;
        setInputLevel(0);
        updateStateIcon(stateListening, false);
        updateStateIcon(stateTranslating, false);
        updateStateIcon(stateSpeaking, false);
        settingsStatusText.textContent = ''; // Clear existing status text
        activityLog.innerHTML = ''; // Clear activity log

        // Reset modal specific elements
        modalReferenceAudioStatus.textContent = ''; // Clear modal status
        modalPhoneticPromptText.innerHTML = '<p class="placeholder">Generating prompt...</p>'; // Reset modal prompt
        if (modalRecordingConfirmation) {
            modalRecordingConfirmation.textContent = ''; // Clear confirmation message
        }
        currentPhoneticPrompt = null; // Clear stored prompt
        referenceAudioBase64 = null; // Clear stored audio
        referenceAudioMimeType = null; // Clear stored mime type
        isModalRecording = false;
        modalRecordBtn.textContent = 'Record';
        modalRecordBtn.classList.remove('recording');
        modalLoadingIndicator.style.display = 'none'; // Hide spinner
        modalLoadingMessage.style.display = 'none'; // Hide message
        modalMicLevelContainer.classList.add('hidden-initial'); // Hide mic level
        modalEmaLevel = 0;
        setModalInputLevel(0);
        modalCurrentWordIndex = 0; // Reset for new recording
    }

    function appendLog(message, type = 'info') {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('p');
        logEntry.classList.add('log-entry', `log-${type}`);
        logEntry.innerHTML = `[${timestamp}] ${escapeHtml(message)}`;
        activityLog.prepend(logEntry); // Add to top
        // Optional: Limit log entries to prevent UI clutter
        while (activityLog.children.length > 50) {
            activityLog.removeChild(activityLog.lastChild);
        }
    }

    // --- WebSocket Handling ---
    function connectWebSocket() {
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            console.log('WebSocket already connected.');
            return;
        }
        try {
            console.log(`Attempting to connect WebSocket to: ${WS_URL}`);
            websocket = new WebSocket(WS_URL);
        } catch (err) {
            console.error('WebSocket constructor error:', err);
            settingsStatusText.textContent = `Invalid WS URL: ${WS_URL}`;
            updateStatus('error', 'WS Error');
            appendLog(`WebSocket constructor error: ${err.message}`, 'error');
            return;
        }

        websocket.onopen = () => {
            console.log('WebSocket connected.');
            updateStatus('on', 'Connected');
            appendLog('WebSocket connected.', 'success');
            // If modal is open, request phonetic prompt now that WS is connected
            if (voiceTrainingModal.style.display === 'flex') {
                modalLoadingIndicator.style.display = 'none'; // Hide spinner
                modalLoadingMessage.style.display = 'none'; // Hide message
                modalRecordBtn.disabled = false; // Enable buttons
                modalChooseFileBtn.disabled = false; // Enable buttons
                modalNotNowBtn.disabled = false; // Enable "Not now" button

                const currentPromptLang = modalInputLanguageSelect.value; // Use modal's language select
                websocket.send(JSON.stringify({
                    type: 'request_phonetic_prompt',
                    language: currentPromptLang
                }));
                modalPhoneticPromptText.innerHTML = '<p class="placeholder">Generating prompt...</p>'; // Update to generating
                appendLog(`Requesting phonetic prompt for ${currentPromptLang.toUpperCase()} after WS connect.`, 'info');
            }
        };

        websocket.onmessage = async (event) => {
            if (event.data instanceof ArrayBuffer) {
                await playAudio(event.data);
            } else {
                try {
                    const data = JSON.parse(event.data);

                    if (data.type === 'audio_level') {
                        setInputLevel(typeof data.level === 'number' ? data.level : 0);
                    } else if (data.type === 'transcription_result') {
                        if (data.transcribed) {
                            transcriptionBox.innerHTML = `<p>${escapeHtml(data.transcribed)}</p>`;
                            sttTime.textContent = `${(data.metrics.stt_time || 0).toFixed(2)}s`;
                            updateStateIcon(stateTranslating, true);
                        }
                    } else if (data.type === 'translation_result') {
                        if (data.translated) {
                            translationBox.innerHTML = `<p>${escapeHtml(data.translated)}</p>`;
                            mtTime.textContent = `${(data.metrics.mt_time || 0).toFixed(2)}s`;
                            updateStateIcon(stateTranslating, false);
                            updateStateIcon(stateSpeaking, true);
                        }
                    } else if (data.type === 'final_metrics') {
                        if (data.metrics) {
                            ttsTime.textContent = `${(data.metrics.tts_time || 0).toFixed(2)}s`;
                            totalTime.textContent = `${(data.metrics.total_latency || 0).toFixed(2)}s`;
                            updateStateIcon(stateSpeaking, false);
                        }
                    } else if (data.type === 'status') {
                        appendLog(data.message || '', 'info');
                        // Specific status updates for reference audio upload in modal
                        if (data.message.includes("Reference audio updated")) {
                            modalReferenceAudioStatus.textContent = data.message;
                            modalReferenceAudioStatus.style.color = 'green';
                        } else if (data.message.includes("Error processing reference audio")) {
                            modalReferenceAudioStatus.textContent = data.message;
                            modalReferenceAudioStatus.style.color = 'red';
                        }
                    } else if (data.type === 'phonetic_prompt_result') {
                        currentPhoneticPrompt = data.prompt_text;
                        // Split prompt into words and wrap each in a span for highlighting
                        modalPhoneticPromptText.innerHTML = currentPhoneticPrompt.split(' ').map((word, index) => 
                            `<span id="prompt-word-${index}">${escapeHtml(word)}</span>`
                        ).join(' ');
                        appendLog('Phonetic prompt generated.', 'info');
                    } else if (data.type === 'modal_transcription_update') {
                        if (data.segments && data.segments.length > 0) {
                            const newWords = data.segments[0].words; // Assuming one segment for simplicity
                            newWords.forEach(wordInfo => {
                                // Find the corresponding word in the phonetic prompt
                                // We need to be careful with word matching due to potential transcription differences
                                // For now, we'll rely on sequential highlighting based on modalCurrentWordIndex
                                if (modalCurrentWordIndex < currentPhoneticPrompt.split(' ').length) {
                                    const promptWordSpan = document.getElementById(`prompt-word-${modalCurrentWordIndex}`);
                                    if (promptWordSpan) {
                                        promptWordSpan.classList.add('highlight');
                                        // Optional: Scroll to the highlighted word
                                        promptWordSpan.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                                    }
                                    modalCurrentWordIndex++;
                                }
                            });
                            appendLog(`Modal transcription update: ${newWords.map(w => w.word).join(' ')}`, 'info');
                        }
                        if (data.is_final) {
                            modalReferenceAudioStatus.textContent = 'Recording finished. Processing...';
                            modalReferenceAudioStatus.style.color = 'green';
                            // After final transcription, we can trigger the upload if needed, or just close modal
                            // For now, we'll assume the backend handles the final processing and then the modal closes.
                            // If we need to upload the recorded audio for voice cloning, we'd need to re-introduce MediaRecorder
                            // or have the backend save the full modal audio buffer.
                            // For this task, we are focusing on real-time highlighting, not modal audio upload.
                        }
                    } else if (data.type === 'models_loading_status') {
                        if (data.loading_started && !data.fully_loaded) {
                            settingsStatusText.textContent = 'Models are loading in the background...';
                            appendLog('Models loading started in background.', 'info');
                            // If modal is open, update its message
                            if (voiceTrainingModal.style.display === 'flex') {
                                modalLoadingMessage.textContent = 'Models are loading in the background. You may train your voice now.';
                                modalLoadingIndicator.style.display = 'none'; // Hide spinner once loading starts
                                modalRecordBtn.disabled = false; // Enable buttons
                                modalChooseFileBtn.disabled = false;
                                modalNotNowBtn.disabled = false; // Enable "Not now" button
                            }
                        } else if (data.fully_loaded) {
                            isInitialized = true; // Set isInitialized to true here
                            settingsStatusText.textContent = 'All models are ready.';
                            appendLog('All models fully loaded.', 'success');
                            startBtn.disabled = false; // Enable main start button
                            initBtn.disabled = true; // Disable init button once fully loaded
                            contentMain.style.display = 'flex'; // Show main content only when models are fully loaded
                            // If modal is open, update its message
                            if (voiceTrainingModal.style.display === 'flex') {
                                modalLoadingMessage.textContent = 'Models are ready. Voice training complete.';
                                modalLoadingIndicator.style.display = 'none';
                                modalRecordBtn.disabled = false;
                                modalChooseFileBtn.disabled = false;
                                modalNotNowBtn.disabled = false; // Re-enable "Not now" button
                                // Optionally close modal if voice training was successful and models are loaded
                                // closeModal();
                            }
                        } else {
                            settingsStatusText.textContent = 'Awaiting model initialization.';
                            appendLog('Awaiting model initialization.', 'info');
                        }
                    } else if (data.status === 'error') {
                        console.error('Server Error:', data.message);
                        appendLog(`Server Error: ${data.message}`, 'error');
                        settingsStatusText.textContent = `Error: ${data.message}`;
                        stopRecordingMain();
                    }
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                    appendLog(`Error parsing WebSocket message: ${error.message}`, 'error');
                }
            }
        };

        websocket.onclose = () => {
            console.log('WebSocket disconnected.');
            updateStatus('off', 'Disconnected');
            appendLog('WebSocket disconnected.', 'warning');
            stopRecordingMain(); // Changed from stopRecording()
            // If modal is open, disable buttons and show waiting message
            if (voiceTrainingModal.style.display === 'flex') {
                modalRecordBtn.disabled = true;
                modalChooseFileBtn.disabled = true;
                modalNotNowBtn.disabled = true; // Disable "Not now" button
                modalLoadingIndicator.style.display = 'block';
                modalLoadingMessage.style.display = 'block';
                modalLoadingMessage.textContent = 'WebSocket disconnected. Waiting for reconnection...';
                modalPhoneticPromptText.innerHTML = '<p class="placeholder">Waiting for server connection to generate prompt...</p>';
            }
        };

        websocket.onerror = (event) => {
            console.error('WebSocket Error Event:', event);
            settingsStatusText.textContent = 'WebSocket connection error. Check console for details.';
            updateStatus('error', 'Error');
            // Attempt to extract more specific error message if available
            const errorMessage = event.message || (event.target && event.target.readyState === WebSocket.CLOSED ? 'Connection closed unexpectedly.' : 'Unknown WebSocket error.');
            appendLog(`WebSocket Error: ${errorMessage}`, 'error');
            stopRecordingMain(); // Changed from stopRecording()
            // If modal is open, disable buttons and show error message
            if (voiceTrainingModal.style.display === 'flex') {
                modalRecordBtn.disabled = true;
                modalChooseFileBtn.disabled = true;
                modalNotNowBtn.disabled = true; // Disable "Not now" button
                modalLoadingIndicator.style.display = 'none'; // Hide spinner on error
                modalLoadingMessage.style.display = 'block';
                modalLoadingMessage.textContent = `WebSocket error: ${errorMessage}`;
                modalPhoneticPromptText.innerHTML = '<p class="placeholder" style="color: red;">WebSocket not open. Cannot generate prompt.</p>';
            }
        };
    }

    function closeWebSocket() {
        if (websocket) {
            websocket.close();
            websocket = null;
        }
    }

    async function playAudio(audioBuffer) {
        if (!audioContext) {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
        }
        try {
            const audioData = await audioContext.decodeAudioData(audioBuffer);
            const source = audioContext.createBufferSource();
            source.buffer = audioData;
            source.connect(audioContext.destination);
            source.start(0);
            appendLog('Playing synthesized audio.', 'success');
        } catch (error) {
            console.error('Error playing audio:', error);
            appendLog(`Error playing audio: ${error.message}`, 'error');
        }
    }

    // --- Audio Recording Handling ---
    // --- Audio Recording Handling (Main UI) ---
    async function startRecordingMain() {
        if (isRecording) return;

        try {
            const devices = await navigator.mediaDevices.enumerateDevices();
            const audioInputDevices = devices.filter(device => device.kind === 'audioinput');
            
            let defaultMicId = null;
            if (audioInputDevices.length > 0) {
                const nonBlackHoleMic = audioInputDevices.find(device => !device.label.toLowerCase().includes('blackhole'));
                defaultMicId = nonBlackHoleMic ? nonBlackHoleMic.deviceId : audioInputDevices[0].deviceId;
                console.log(`[UI] Using audio input device: ${nonBlackHoleMic ? nonBlackHoleMic.label : audioInputDevices[0].label} (ID: ${defaultMicId})`);
            } else {
                console.warn("[UI] No audio input devices found.");
                settingsStatusText.textContent = "No microphone found. Please connect one.";
                updateStatus('error', 'Mic Error');
                return;
            }

            mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    deviceId: defaultMicId ? { exact: defaultMicId } : undefined,
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false
                }
            });

            audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: SAMPLE_RATE
            });

            const source = audioContext.createMediaStreamSource(mediaStream);
            audioProcessor = audioContext.createScriptProcessor(BUFFER_SIZE, 1, 1);
            audioProcessor.onaudioprocess = (event) => {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    const audioData = event.inputBuffer.getChannelData(0);
                    const audioClone = new Float32Array(audioData);
                    websocket.send(audioClone.buffer);
                    const now = Date.now();
                    if (now - lastLogAt > LOG_THROTTLE_MS) {
                        console.debug(`Sent ${audioClone.length} audio samples`);
                        lastLogAt = now;
                    }
                }
            };

            source.connect(audioProcessor);
            audioProcessor.connect(audioContext.destination);

            isRecording = true;
            updateStatus('on', 'Listening...');
            startBtn.disabled = true;
            stopBtn.disabled = false;
            updateStateIcon(stateListening, true);

            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(JSON.stringify({ type: 'start' }));
            }
        } catch (error) {
            console.error('Error accessing microphone:', error);
            settingsStatusText.textContent = `Microphone access denied: ${error.message}`;
            updateStatus('error', 'Mic Error');
            stopRecordingMain();
        }
    }

    function stopRecordingMain() {
        if (!isRecording) return;

        isRecording = false;
        updateStatus('off', 'Ready');
        startBtn.disabled = false;
        stopBtn.disabled = true;
        updateStateIcon(stateListening, false);
        updateStateIcon(stateTranslating, false);
        updateStateIcon(stateSpeaking, false);
        resetUI();

        if (mediaStream) {
            mediaStream.getTracks().forEach(track => track.stop());
            mediaStream = null;
        }
        if (audioProcessor) {
            try {
                audioProcessor.disconnect();
            } catch (_) { /* ignore */ }
            audioProcessor = null;
        }
        if (audioContext) {
            audioContext.close();
            audioContext = null;
        }

        if (websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.send(JSON.stringify({ type: 'stop' }));
        }
    }

    // --- Audio Recording Handling (Modal) ---
    let mediaRecorder;
    let audioChunks = [];

    async function startModalRecording() {
        if (isModalRecording) {
            stopModalRecording();
            return;
        }

        try {
            modalMediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            modalAudioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
            const source = modalAudioContext.createMediaStreamSource(modalMediaStream);
            modalAudioProcessor = modalAudioContext.createScriptProcessor(BUFFER_SIZE, 1, 1);

            modalAudioProcessor.onaudioprocess = (event) => {
                const audioData = event.inputBuffer.getChannelData(0);
                // Calculate RMS for mic level visualization
                let sumSquares = 0;
                for (const sample of audioData) {
                    sumSquares += sample * sample;
                }
                const rms = Math.sqrt(sumSquares / audioData.length);
                setModalInputLevel(rms);

                // Send audio data to WebSocket for real-time processing
                if (websocket && websocket.readyState === WebSocket.OPEN && isModalRecording) {
                    const audioClone = new Float32Array(audioData);
                    websocket.send(audioClone.buffer); // Send raw audio bytes
                }
            };

            source.connect(modalAudioProcessor);
            modalAudioProcessor.connect(modalAudioContext.destination); // Connect to destination to ensure onaudioprocess fires

            // No longer using MediaRecorder for chunk sending, but for fallback or if needed later
            // mediaRecorder = new MediaRecorder(modalMediaStream);
            // audioChunks = [];

            // mediaRecorder.ondataavailable = event => {
            //     audioChunks.push(event.data);
            // };

            // mediaRecorder.onstop = async () => {
            //     // This part is now handled by sending chunks directly
            // };

            // mediaRecorder.start(); // No longer needed for chunk sending
            isModalRecording = true;
            modalRecordBtn.textContent = 'Stop Recording';
            modalRecordBtn.classList.add('recording');
            modalReferenceAudioStatus.textContent = 'Recording...';
            modalReferenceAudioStatus.style.color = 'orange';
            modalMicLevelContainer.classList.remove('hidden-initial'); // Show mic level
            appendLog('Started modal recording.', 'info');

            // Send a 'start_modal_recording' message to the backend
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(JSON.stringify({ type: 'start_modal_recording', language: modalInputLanguageSelect.value }));
            }

        } catch (error) {
            console.error('Error accessing microphone for modal recording:', error);
            modalReferenceAudioStatus.textContent = `Mic access denied: ${error.message}`;
            modalReferenceAudioStatus.style.color = 'red';
            appendLog(`Error accessing microphone for modal recording: ${error.message}`, 'error');
        }
    }

    function stopModalRecording() {
        if (!isModalRecording) return;

        isModalRecording = false;
        modalRecordBtn.textContent = 'Record';
        modalRecordBtn.classList.remove('recording');
        appendLog('Stopped modal recording.', 'info');

        // Send a 'stop_modal_recording' message to the backend
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.send(JSON.stringify({ type: 'stop_modal_recording' }));
        }

        // if (mediaRecorder && mediaRecorder.state !== 'inactive') { // No longer using MediaRecorder for chunk sending
        //     mediaRecorder.stop();
        // }
        if (modalMediaStream) {
            modalMediaStream.getTracks().forEach(track => track.stop());
            modalMediaStream = null;
        }
        if (modalAudioProcessor) {
            try {
                modalAudioProcessor.disconnect();
            } catch (_) { /* ignore */ }
            modalAudioProcessor = null;
        }
        if (modalAudioContext) {
            modalAudioContext.close();
            modalAudioContext = null;
        }
        modalMicLevelContainer.classList.add('hidden-initial'); // Hide mic level
        modalEmaLevel = 0;
        setModalInputLevel(0);
    }

    async function uploadReferenceAudioToBackend() {
        if (!websocket || websocket.readyState !== WebSocket.OPEN) {
            modalReferenceAudioStatus.textContent = 'Not connected to server. Cannot upload audio. Please wait for models to load and WebSocket to connect.';
            modalReferenceAudioStatus.style.color = 'red';
            appendLog('Not connected to server. Cannot upload reference audio.', 'warning');
            return;
        }

        if (!referenceAudioBase64 || !referenceAudioMimeType) {
            modalReferenceAudioStatus.textContent = 'No audio data to upload.';
            modalReferenceAudioStatus.style.color = 'red';
            return;
        }

        modalReferenceAudioStatus.textContent = 'Uploading...';
        modalReferenceAudioStatus.style.color = 'inherit';

        try {
            const uploadPayload = {
                type: 'upload_reference_audio',
                audio_data_base64: referenceAudioBase64,
                mime_type: referenceAudioMimeType
            };

            if (currentPhoneticPrompt) {
                uploadPayload.provided_transcription = currentPhoneticPrompt;
            }

            websocket.send(JSON.stringify(uploadPayload));
            appendLog(`Reference audio uploaded.`, 'info');
            modalReferenceAudioStatus.textContent = `Reference audio sent.`;
            modalReferenceAudioStatus.style.color = 'green';
            
            // Display the processing message
            modalLoadingIndicator.style.display = 'block';
            modalLoadingMessage.style.display = 'block';
            modalLoadingMessage.textContent = "Your voice is being processed, please wait for other modules to load ~1 minute. You may exit this window now.";
            modalRecordBtn.disabled = true; // Disable buttons during processing
            modalChooseFileBtn.disabled = true;
            modalNotNowBtn.disabled = true; // Disable "Not now" button as well

            // The modal will be closed by the backend's 'status' message or 'models_loaded_status'
            // when the reference audio processing is complete and models are fully loaded.
            // For now, we'll keep it open with the message.
            // closeModal(); // Do not close immediately, wait for backend confirmation
        } catch (error) {
            console.error('Error uploading reference audio:', error);
            modalReferenceAudioStatus.textContent = `Error uploading: ${error.message}`;
            modalReferenceAudioStatus.style.color = 'red';
            appendLog(`Error uploading reference audio: ${error.message}`, 'error');
        }
    }

    // --- Modal Functions ---
    function openModal() {
        voiceTrainingModal.style.display = 'flex';
        // Always show a loading message initially
        modalPhoneticPromptText.innerHTML = '<p class="placeholder">Waiting for server connection to generate prompt...</p>';
        modalLoadingIndicator.style.display = 'block'; // Show spinner
        modalLoadingMessage.style.display = 'block'; // Show message
        modalLoadingMessage.textContent = 'Please wait until models preload before you record your voice.';
        modalRecordBtn.disabled = true; // Disable buttons
        modalChooseFileBtn.disabled = true; // Disable buttons
        appendLog('Voice training modal opened. Waiting for WebSocket connection.', 'info');

        // Request phonetic prompt only if WS is already open
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            const currentPromptLang = modalInputLanguageSelect.value; // Use modal's language select
            websocket.send(JSON.stringify({
                type: 'request_phonetic_prompt',
                language: currentPromptLang
            }));
            modalPhoneticPromptText.innerHTML = '<p class="placeholder">Generating prompt...</p>'; // Update to generating
            appendLog(`Requesting phonetic prompt for ${currentPromptLang.toUpperCase()}.`, 'info');
        } else {
            // If WS is not open, the onopen handler will request the prompt once connected.
            // For now, just log and display waiting message.
            appendLog('WebSocket not open. Will request phonetic prompt after connection.', 'info');
        }
    }

    function closeModal() {
        voiceTrainingModal.style.display = 'none';
        modalReferenceAudioStatus.textContent = ''; // Clear modal status
        modalPhoneticPromptText.innerHTML = '<p class="placeholder">Generating prompt...</p>'; // Reset modal prompt
        if (modalRecordingConfirmation) {
            modalRecordingConfirmation.textContent = ''; // Clear confirmation message
        }
        currentPhoneticPrompt = null; // Clear stored prompt
        referenceAudioBase64 = null; // Clear stored audio
        referenceAudioMimeType = null; // Clear stored mime type
        isModalRecording = false;
        modalRecordBtn.textContent = 'Record';
        modalRecordBtn.classList.remove('recording');
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        // Reset word highlights
        if (currentPhoneticPrompt) {
            const words = currentPhoneticPrompt.split(' ');
            for (let i = 0; i < words.length; i++) {
                const wordSpan = document.getElementById(`prompt-word-${i}`);
                if (wordSpan) {
                    wordSpan.classList.remove('highlight');
                }
            }
        }
        // Reset modal specific UI elements
        modalLoadingIndicator.style.display = 'none';
        modalLoadingMessage.style.display = 'none';
        modalMicLevelContainer.classList.add('hidden-initial');
        modalEmaLevel = 0;
        setModalInputLevel(0);
        modalCurrentWordIndex = 0; // Reset for new modal session
    }


    // --- Helpers & Events ---
    initBtn.addEventListener('click', async () => {
        if (isInitialized) {
            settingsStatusText.textContent = 'Pipeline already initialized.';
            return;
        }
        initBtn.disabled = true;
        updateStatus('prepping', 'Initializing Models...');
        appendLog('Initializing models...', 'info');
        settingsStatusText.textContent = 'Loading models (this may take ~1 minute on first run)...';
        
        // Connect WebSocket immediately
        connectWebSocket();

        // Trigger backend initialization (which starts background loading)
        try {
            const sourceLang = inputLanguageSelect.value;
            const targetLang = outputLanguageSelect.value;
            const ttsModel = ttsModelSelect.value;

            appendLog(`Attempting to initialize pipeline with: Source=${sourceLang.toUpperCase()}, Target=${targetLang.toUpperCase()}, TTS Model=${ttsModel}.`, 'info');

            const response = await fetch(`${API_URL}/initialize?source_lang=${sourceLang}&target_lang=${targetLang}&tts_model_choice=${ttsModel}`, { method: 'POST' });
            const data = await response.json();

            if (data.status === 'success') {
                // isInitialized will be set to true when models_fully_loaded is received
                // startBtn will be enabled when models_fully_loaded is received
                settingsStatusText.textContent = data.message;
                appendLog(`Pipeline initialization triggered: ${data.message}`, 'success');
                contentMain.style.display = 'flex'; // Show main content
                openModal(); // Open the voice training modal after initialization is triggered
            } else {
                settingsStatusText.textContent = `Initialization failed: ${data.message}`;
                updateStatus('error', 'Init Error');
                appendLog(`Initialization failed: ${data.message}`, 'error');
                initBtn.disabled = false;
            }
        } catch (error) {
            console.error('Initialization API error:', error);
            settingsStatusText.textContent = `Initialization failed: ${error.message}`;
            updateStatus('error', 'Init Error');
            appendLog(`Initialization API error: ${error.message}`, 'error');
            initBtn.disabled = false;
        }
    });

    // No longer a separate "Train Voice" button, functionality integrated into initBtn
    // const trainVoiceBtn = document.getElementById('trainVoiceBtn');
    // trainVoiceBtn.addEventListener('click', () => {
    //     openModal();
    // });

    modalNotNowBtn.addEventListener('click', () => {
        // Close modal and proceed. Model loading is already triggered by initBtn.
        closeModal();
    });

    modalRecordBtn.addEventListener('click', startModalRecording); // New record button handler

    modalChooseFileBtn.addEventListener('click', () => { // New choose file button handler
        modalReferenceAudioUpload.click(); // Trigger file input click
    });

    modalReferenceAudioUpload.addEventListener('change', async (event) => {
        if (!websocket || websocket.readyState !== WebSocket.OPEN) {
            modalReferenceAudioStatus.textContent = 'Not connected to server. Cannot upload audio. Please wait for models to load and WebSocket to connect.';
            modalReferenceAudioStatus.style.color = 'red';
            appendLog('Not connected to server. Cannot upload reference audio.', 'warning');
            return;
        }

        const file = event.target.files[0];
        if (!file) {
            modalReferenceAudioStatus.textContent = 'No file selected.';
            modalReferenceAudioStatus.style.color = 'red';
            return;
        }

        modalReferenceAudioStatus.textContent = 'Uploading...';
        modalReferenceAudioStatus.style.color = 'inherit';

        try {
            const reader = new FileReader();
            reader.readAsDataURL(file);
            reader.onload = () => {
                referenceAudioBase64 = reader.result.split(',')[1];
                referenceAudioMimeType = file.type;
                uploadReferenceAudioToBackend(); // Use the new function
            };
            reader.onerror = (error) => {
                console.error('Error reading file:', error);
                modalReferenceAudioStatus.textContent = `Error reading file: ${error.message}`;
                modalReferenceAudioStatus.style.color = 'red';
                appendLog(`Error reading reference audio file: ${error.message}`, 'error');
            };
        } catch (error) {
            console.error('Error uploading reference audio:', error);
            modalReferenceAudioStatus.textContent = `Error uploading: ${error.message}`;
            modalReferenceAudioStatus.style.color = 'red';
            appendLog(`Error uploading reference audio: ${error.message}`, 'error');
        }
    });

    startBtn.addEventListener('click', startRecordingMain); // Renamed for clarity
    stopBtn.addEventListener('click', stopRecordingMain); // Renamed for clarity

    function sendConfigUpdate() {
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            const config = {
                type: 'config_update',
                source_lang: inputLanguageSelect.value,
                target_lang: outputLanguageSelect.value,
                tts_model_choice: ttsModelSelect.value
            };
            websocket.send(JSON.stringify(config));
            appendLog(`Configuration updated: Source=${config.source_lang.toUpperCase()}, Target=${config.target_lang.toUpperCase()}, TTS Model=${config.tts_model_choice}.`, 'info');
            settingsStatusText.textContent = 'Configuration updated.'; // Keep this for immediate feedback
        } else {
            settingsStatusText.textContent = 'Not connected to server. Cannot update config.';
            appendLog('Not connected to server. Cannot update config.', 'warning');
        }
    }

    inputLanguageSelect.addEventListener('change', (event) => {
        inputLanguageBadge.textContent = event.target.value.toUpperCase();
        appendLog(`Input language changed to ${event.target.value.toUpperCase()}.`, 'info');
        sendConfigUpdate();
        // If modal is open, update the prompt language
        if (voiceTrainingModal.style.display === 'flex' && websocket && websocket.readyState === WebSocket.OPEN) {
            const currentPromptLang = modalInputLanguageSelect.value;
            websocket.send(JSON.stringify({
                type: 'request_phonetic_prompt',
                language: currentPromptLang
            }));
            modalPhoneticPromptText.innerHTML = '<p class="placeholder">Generating prompt...</p>';
            appendLog(`Requesting new phonetic prompt for ${currentPromptLang.toUpperCase()} due to input language change.`, 'info');
        }
    });

    outputLanguageSelect.addEventListener('change', (event) => {
        outputLanguageBadge.textContent = event.target.value.toUpperCase();
        appendLog(`Output language changed to ${event.target.value.toUpperCase()}.`, 'info');
        sendConfigUpdate();
    });

    ttsModelSelect.addEventListener('change', (event) => {
        appendLog(`TTS Model changed to ${event.target.value}.`, 'info');
        sendConfigUpdate();
    });

    modalInputLanguageSelect.addEventListener('change', (event) => {
        appendLog(`Modal prompt language changed to ${event.target.value.toUpperCase()}.`, 'info');
        if (websocket && websocket.readyState === WebSocket.OPEN) {
            websocket.send(JSON.stringify({
                type: 'request_phonetic_prompt',
                language: event.target.value
            }));
            modalPhoneticPromptText.innerHTML = '<p class="placeholder">Generating prompt...</p>';
            appendLog(`Requesting new phonetic prompt for ${event.target.value.toUpperCase()}.`, 'info');
        }
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
    updateStatus('off', 'Awaiting Initialization');
    startBtn.disabled = true;
    stopBtn.disabled = true;
});
