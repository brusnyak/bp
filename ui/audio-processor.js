class AudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.sampleRate = 16000; // Match backend sample rate
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];

    if (input.length > 0) {
      const inputChannelData = input[0]; // Assuming mono input
      this.port.postMessage(inputChannelData); // Send audio data to main thread
    }

    // Pass through audio to output (optional, depending on desired behavior)
    for (let channel = 0; channel < input.length; ++channel) {
      output[channel].set(input[channel]);
    }

    return true;
  }
}

registerProcessor("audio-processor", AudioProcessor);
