import os
import torch
from TTS.api import TTS

class XTTS_TTS:
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2", device="auto"):
        # Determine the actual device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        self.model_name = model_name
        self.tts = None
        self.speaker_wav_path = None # Path to the speaker reference audio

    async def load_model(self):
        if self.tts is None:
            print(f"Loading XTTS v2 model: {self.model_name} to device: {self.device}")
            # TTS().to() expects a torch.device object or a string like "cpu", "cuda", "mps"
            self.tts = TTS(model_name=self.model_name, progress_bar=True).to(torch.device(self.device))
            print("XTTS v2 model loaded.")

    def set_speaker_wav(self, speaker_wav_path):
        if not os.path.exists(speaker_wav_path):
            raise FileNotFoundError(f"Speaker WAV file not found at: {speaker_wav_path}")
        self.speaker_wav_path = speaker_wav_path
        print(f"XTTS v2 speaker reference set to: {self.speaker_wav_path}")

    async def synthesize(self, text: str, language: str) -> bytes:
        if self.tts is None:
            raise RuntimeError("XTTS v2 model not loaded. Call load_model() first.")
        if self.speaker_wav_path is None:
            raise RuntimeError("Speaker WAV not set. Call set_speaker_wav() first.")

        print(f"Synthesizing speech with XTTS v2 for text: '{text}' in language: {language}")
        
        # Use synthesize_stream to get chunks
        audio_chunks = []
        for chunk in self.tts.synthesize_stream(
            text=text,
            speaker_wav=self.speaker_wav_path,
            language=language,
            split_sentences=True # XTTS v2 handles sentence splitting internally
        ):
            audio_chunks.append(chunk)
        
        # Concatenate all chunks into a single numpy array
        if not audio_chunks:
            return b"" # Return empty bytes if no audio was generated
        
        wav = torch.cat(audio_chunks, dim=0).cpu().numpy()

        # The tts method returns a numpy array, convert to bytes
        # Assuming 16kHz, 16-bit PCM for WAV output
        # For real-time streaming, you might want to return raw PCM bytes
        # For now, let's return a simple WAV file in memory
        import io
        import soundfile as sf
        
        buffer = io.BytesIO()
        sf.write(buffer, wav, samplerate=self.tts.synthesizer.output_sample_rate, format='WAV')
        buffer.seek(0)
        return buffer.getvalue()

    async def synthesize_stream(self, text: str, language: str):
        if self.tts is None:
            raise RuntimeError("XTTS v2 model not loaded. Call load_model() first.")
        if self.speaker_wav_path is None:
            raise RuntimeError("Speaker WAV not set. Call set_speaker_wav() first.")

        print(f"Streaming speech with XTTS v2 for text: '{text}' in language: {language}")
        
        # The synthesize_stream method yields audio chunks
        for chunk in self.tts.synthesize_stream(
            text=text,
            speaker_wav=self.speaker_wav_path,
            language=language,
            split_sentences=True
        ):
            # Convert each chunk (torch tensor) to bytes
            # Assuming 16kHz, 16-bit PCM for WAV output
            buffer = io.BytesIO()
            sf.write(buffer, chunk.cpu().numpy(), samplerate=self.tts.synthesizer.output_sample_rate, format='WAV')
            buffer.seek(0)
            yield buffer.getvalue()

if __name__ == '__main__':
    # Example usage
    async def main():
        xtts_model = XTTS_TTS(device="cpu") # Use "cuda" if GPU is available
        await xtts_model.load_model()

        # You need a speaker reference audio file for XTTS v2
        # For testing, you can use a sample WAV file
        # Make sure to replace 'path/to/your/speaker.wav' with an actual path
        # For example, you can use the 'Voice-Training.wav' from the test folder
        speaker_wav_file = "../../test/Voice-Training.wav" 
        xtts_model.set_speaker_wav(speaker_wav_file)

        text_to_synthesize = "Hello, this is a test of XTTS v2 speech synthesis."
        language_to_use = "en" # or "sk", "cs", etc.

        audio_bytes = await xtts_model.synthesize(text_to_synthesize, language_to_use)

        with open("xtts_output.wav", "wb") as f:
            f.write(audio_bytes)
        print("XTTS v2 synthesized audio saved to xtts_output.wav")

    import asyncio
    asyncio.run(main())
