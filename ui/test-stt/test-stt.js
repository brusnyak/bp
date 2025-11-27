// STT Test Page JavaScript
// Isolated testing for Speech-to-Text functionality

const WS_URL = `wss://${window.location.host}/ws`;
let websocket = null;
let audioContext = null;
let audioStream = null;
let audioWorkletNode = null;
let analyser = null;
let micSource = null;

const initBtn = document.getElementById('initBtn');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusDiv = document.getElementById('status');
const transcriptContainer = document.getElementById('transcriptContainer');
const micLevelFill = document.getElementById('micLevelFill');

let isListening = false;
let transcriptCount = 0;

// Initialize models
initBtn.addEventListener('click', async () => {
    console.log('Initializing models...');
    setStatus('Initializing models...', 'processing');

    try {
        // Use relative path - works when accessed via https://localhost:8000/ui/test-stt/test-stt.html
        const response = await fetch('/initialize?source_lang=en&target_lang=sk&tts_model_choice=piper', {
            method: 'POST'
        });

        const result = await response.json();
        console.log('Initialization result:', result);

        if (result.status === 'success') {
            setStatus('Models initialized. Ready to start.', 'idle');
            initBtn.disabled = true;
            startBtn.disabled = false;
        } else {
            setStatus('Initialization failed: ' + result.message, 'idle');
        }
    } catch (error) {
        console.error('Initialization error:', error);
        setStatus('Initialization error: ' + error.message, 'idle');
    }
});

// Start listening
startBtn.addEventListener('click', async () => {
    console.log('Starting audio capture...');
    setStatus('Starting microphone...', 'processing');

    try {
        // Get microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: true
            }
        });

        console.log('Microphone access granted');

        // Create audio context
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000
        });

        console.log('AudioContext created with sample rate:', audioContext.sampleRate);

        // Load AudioWorklet
        await audioContext.audioWorklet.addModule('/ui/live-speech/audio-processor.js');
        console.log('AudioWorklet loaded');

        // Create audio processing chain
        micSource = audioContext.createMediaStreamSource(audioStream);
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;

        audioWorkletNode = new AudioWorkletNode(audioContext, 'audio-processor');

        // Handle audio data from worklet
        audioWorkletNode.port.onmessage = (event) => {
            const audioData = event.data;
            const audioDataFloat32 = new Float32Array(audioData);

            // Send to backend via WebSocket
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.send(audioDataFloat32.buffer);
            }
        };

        // Connect audio chain
        // Create GainNode to amplify mic input (matching live-script.js)
        const gainNode = audioContext.createGain();
        gainNode.gain.value = 4.0; // 12dB boost
        console.log('Added GainNode with gain:', gainNode.gain.value);

        // Connect audio chain: Source -> Gain -> Analyser -> Worklet
        micSource.connect(gainNode);
        gainNode.connect(analyser);
        analyser.connect(audioWorkletNode);

        // Start mic level visualization
        startMicLevelVisualization();

        // Connect WebSocket
        connectWebSocket();

        isListening = true;
        setStatus('Listening... Speak now!', 'listening');
        startBtn.disabled = true;
        stopBtn.disabled = false;

    } catch (error) {
        console.error('Error starting audio capture:', error);
        setStatus('Error: ' + error.message, 'idle');
    }
});

// Stop listening
stopBtn.addEventListener('click', () => {
    console.log('Stopping audio capture...');
    stopAudioCapture();
    setStatus('Stopped', 'idle');
    startBtn.disabled = false;
    stopBtn.disabled = true;
});

// Connect WebSocket
function connectWebSocket() {
    console.log('Connecting to WebSocket:', WS_URL);
    websocket = new WebSocket(WS_URL);

    websocket.onopen = () => {
        console.log('WebSocket connected');
    };

    websocket.onmessage = (event) => {
        if (typeof event.data === 'string') {
            const message = JSON.parse(event.data);
            console.log('Received message:', message);

            if (message.type === 'transcription_result') {
                addTranscript(message.transcribed, 'en');
            } else if (message.type === 'audio_level') {
                // Ignore for now
            }
        }
    };

    websocket.onerror = (error) => {
        console.error('WebSocket error:', error);
        setStatus('WebSocket error', 'idle');
    };

    websocket.onclose = () => {
        console.log('WebSocket closed');
    };
}

// Stop audio capture
function stopAudioCapture() {
    if (audioWorkletNode) {
        audioWorkletNode.disconnect();
        audioWorkletNode = null;
    }

    if (analyser) {
        analyser.disconnect();
        analyser = null;
    }

    if (micSource) {
        micSource.disconnect();
        micSource = null;
    }

    if (audioStream) {
        audioStream.getTracks().forEach(track => track.stop());
        audioStream = null;
    }

    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }

    if (websocket) {
        websocket.close();
        websocket = null;
    }

    isListening = false;
    micLevelFill.style.width = '0%';
}

// Add transcript to UI
function addTranscript(text, language) {
    transcriptCount++;

    // Remove empty state if present
    const emptyState = transcriptContainer.querySelector('.empty-state');
    if (emptyState) {
        emptyState.remove();
    }

    // Create transcript item
    const item = document.createElement('div');
    item.className = 'transcript-item';

    const time = new Date().toLocaleTimeString();
    const timeDiv = document.createElement('div');
    timeDiv.className = 'transcript-time';
    timeDiv.textContent = `#${transcriptCount} - ${time} - ${language}`;

    const textDiv = document.createElement('div');
    textDiv.className = 'transcript-text';
    textDiv.textContent = text;

    item.appendChild(timeDiv);
    item.appendChild(textDiv);

    // Add to container (newest at top)
    transcriptContainer.insertBefore(item, transcriptContainer.firstChild);

    console.log(`Transcript #${transcriptCount}:`, text);
}

// Set status
function setStatus(message, type) {
    statusDiv.textContent = 'Status: ' + message;
    statusDiv.className = 'status ' + type;
}

// Mic level visualization
function startMicLevelVisualization() {
    const dataArray = new Uint8Array(analyser.frequencyBinCount);

    function updateLevel() {
        if (!isListening || !analyser) return;

        analyser.getByteTimeDomainData(dataArray);

        // Calculate RMS
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
            const normalized = (dataArray[i] - 128) / 128;
            sum += normalized * normalized;
        }
        const rms = Math.sqrt(sum / dataArray.length);

        // Scale to percentage (0-100%)
        const level = Math.min(100, rms * 200);
        micLevelFill.style.width = level + '%';

        requestAnimationFrame(updateLevel);
    }

    updateLevel();
}

console.log('STT Test Page loaded');
