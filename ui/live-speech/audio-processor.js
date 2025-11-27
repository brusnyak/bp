class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.sampleRate = 16000; // Match backend sample rate
    this.bufferSize = 480; // 30ms at 16kHz (16000 * 0.030)
    this.audioBuffer = new Float32Array(this.bufferSize);
    this.bufferIndex = 0;

    this.port.onmessage = (event) => {
      // Handle messages from the main thread
      if (event.data === 'clear_buffer') {
        // Clear the buffer to prevent sending buffered TTS audio
        this.bufferIndex = 0;
        this.audioBuffer.fill(0);
        console.log('[AudioWorklet] Buffer cleared');
      }
    };
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    // Assuming mono input, take the first channel
    const inputChannelData = input[0];

    if (inputChannelData) {
      for (let i = 0; i < inputChannelData.length; i++) {
        this.audioBuffer[this.bufferIndex++] = inputChannelData[i];

        if (this.bufferIndex === this.bufferSize) {
          // Buffer is full, post message to main thread
          this.port.postMessage(this.audioBuffer);
          this.bufferIndex = 0; // Reset buffer index
        }
      }
    }

    // The AudioWorkletNode is not connected to the audioContext.destination in script.js,
    // so there's no need to pass through or clear the output buffer.
    // Removing this section to prevent "Cannot assign to read only property '0'" error.
    // If pass-through is ever needed, outputs[channel].set() should be used with a writable buffer.

    return true;
  }
}

registerProcessor("audio-processor", AudioProcessor);
