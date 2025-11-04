import time
import soundfile as sf
import pydub
from pydub import AudioSegment
from pydub.silence import split_on_silence
import numpy as np
import os
from jiwer import wer
from sacrebleu import corpus_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function execution timed out")

class ModelTestFramework:
    def __init__(self, audio_path, transcript_path=None, translation_path=None):
        # Download NLTK data for METEOR if not already present
        self._download_nltk_data()

        self.audio_path = audio_path
        self.transcript_path = transcript_path
        self.translation_path = translation_path
        self.audio_data, self.samplerate = self._load_audio_and_resample(audio_path)
        self.reference_transcript = self._load_text(transcript_path) if transcript_path else None
        self.reference_translation = self._load_text(translation_path) if translation_path else None

    def _download_nltk_data(self):
        """Downloads necessary NLTK data if not already present."""
        # Ensure 'averaged_perceptron_tagger' is downloaded for spacy-pkuseg if needed
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger', quiet=True)
            print("NLTK resource 'averaged_perceptron_tagger' ensured.")
        for resource in ['wordnet', 'punkt', 'punkt_tab']: # Added 'punkt_tab'
            try:
                nltk.data.find(f'tokenizers/{resource}') # Corrected path for nltk.data.find
            except LookupError:
                print(f"Attempting to download NLTK resource: {resource}...")
                nltk.download(resource, quiet=True)
                print(f"NLTK resource '{resource}' ensured.")

    def _load_audio_and_resample(self, path, target_samplerate=16000):
        """Loads audio file, resamples to target_samplerate, and returns data and samplerate."""
        data, samplerate = sf.read(path)

        if samplerate != target_samplerate:
            print(f"Resampling audio from {samplerate}Hz to {target_samplerate}Hz.")
            # Convert float audio data to int16 for pydub
            # Scale to int16 range and convert type
            int_data = (data * (2**15 - 1)).astype(np.int16)
            
            audio_segment = AudioSegment(
                int_data.tobytes(),
                frame_rate=samplerate,
                sample_width=int_data.dtype.itemsize, # Should be 2 for int16
                channels=1 if len(int_data.shape) == 1 else int_data.shape[1]
            )
            audio_segment = audio_segment.set_frame_rate(target_samplerate)
            # Convert back to float32 for models
            data = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / (2**15 - 1)
            samplerate = target_samplerate
        else:
            data = data.astype(np.float32) # Ensure float32 for models

        return data, samplerate

    def _load_text(self, path):
        """Loads text file content."""
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def measure_latency(self, func, *args, save_result_func=None, output_path=None, timeout=None, **kwargs):
        """
        Measures the execution time of a function with an optional timeout.
        If save_result_func is provided, it will be called with the result and output_path.
        """
        result = None
        latency = -1.0 # Indicate timeout or error if not updated

        if timeout is not None:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)

        try:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            latency = end_time - start_time

            if save_result_func and output_path:
                save_result_func(result, output_path)
        except TimeoutException:
            print(f"Function '{func.__name__}' timed out after {timeout} seconds.")
            result = None # Indicate failure due to timeout
            latency = float('inf') # Indicate infinite latency for timeout
        finally:
            if timeout is not None:
                signal.alarm(0) # Disable the alarm

        return result, latency

    def evaluate_stt(self, predicted_transcript):
        """Evaluates STT accuracy using Word Error Rate (WER)."""
        if not self.reference_transcript:
            print("Warning: No reference transcript provided for STT evaluation.")
            return None
        
        error = wer(self.reference_transcript, predicted_transcript)
        print(f"Reference STT: {self.reference_transcript}")
        print(f"Predicted STT: {predicted_transcript}")
        print(f"STT Word Error Rate (WER): {error:.4f}")
        return {"reference": self.reference_transcript, "predicted": predicted_transcript, "wer": error}

    def evaluate_mt(self, predicted_translation):
        """Evaluates MT accuracy using BLEU and METEOR scores."""
        if not self.reference_translation:
            print("Warning: No reference translation provided for MT evaluation.")
            return None
        
        # BLEU score
        # sacrebleu expects a list of hypotheses (list of strings) and a list of references (list of list of strings)
        bleu = corpus_bleu([str(predicted_translation)], [[str(self.reference_translation)]]).score
        
        # METEOR score
        # NLTK's meteor_score expects a list of references (list of tokenized strings) and a tokenized hypothesis string
        # Tokenize the sentences for METEOR
        tokenized_reference = nltk.word_tokenize(str(self.reference_translation))
        tokenized_predicted = nltk.word_tokenize(str(predicted_translation))
        meteor = meteor_score([tokenized_reference], tokenized_predicted)

        print(f"Reference MT: {self.reference_translation}")
        print(f"Predicted MT: {predicted_translation}")
        print(f"MT BLEU Score: {bleu:.2f}")
        print(f"MT METEOR Score: {meteor:.4f}")
        return {"reference": self.reference_translation, "predicted": predicted_translation, "bleu": bleu, "meteor": meteor}

    def evaluate_tts(self, audio_output_path):
        """Evaluates TTS quality (subjective). Placeholder for objective metrics."""
        print(f"TTS audio saved to: {audio_output_path}")
        # Subjective evaluation by listening to the output
        return {"output_path": audio_output_path}

    def split_audio_on_silence(self, min_silence_len=1000, silence_thresh=-40, keep_silence=500):
        """
        Splits audio into chunks based on silence.
        Returns a list of AudioSegment objects.
        """
        audio = AudioSegment(
            self.audio_data.tobytes(),
            frame_rate=self.samplerate,
            sample_width=self.audio_data.dtype.itemsize,
            channels=1 if len(self.audio_data.shape) == 1 else self.audio_data.shape[1]
        )
        chunks = split_on_silence(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh,
            keep_silence=keep_silence
        )
        return chunks

    def save_audio_chunk(self, chunk, output_filename="temp_chunk.wav"):
        """Saves an AudioSegment chunk to a WAV file."""
        chunk.export(output_filename, format="wav")
        return output_filename

    def cleanup_temp_files(self, *filenames):
        """Removes temporary files."""
        for filename in filenames:
            if os.path.exists(filename):
                os.remove(filename)

# Example Usage (for demonstration, will be moved to specific test files)
if __name__ == "__main__":
    # Create dummy files for testing the framework
    if not os.path.exists("test"):
        os.makedirs("test")
    
    dummy_audio_path = "test/dummy_audio.wav"
    dummy_transcript_path = "test/dummy_transcript.txt"
    dummy_translation_path = "test/dummy_translation.txt"

    # Generate a dummy WAV file
    samplerate = 16000
    duration = 1  # seconds
    frequency = 440  # Hz
    t = np.linspace(0., duration, int(samplerate * duration), endpoint=False)
    amplitude = np.iinfo(np.int16).max * 0.5
    data = amplitude * np.sin(2. * np.pi * frequency * t)
    sf.write(dummy_audio_path, data.astype(np.int16), samplerate)

    with open(dummy_transcript_path, "w") as f:
        f.write("This is a dummy transcript.")
    with open(dummy_translation_path, "w") as f:
        f.write("Toto je fiktívny prepis.")

    framework = ModelTestFramework(
        audio_path=dummy_audio_path,
        transcript_path=dummy_transcript_path,
        translation_path=dummy_translation_path
    )

    # Test latency measurement
    def dummy_stt_model(audio_data):
        time.sleep(0.1) # Simulate processing time
        return "This is a dummy transcript."

    predicted_stt, stt_latency = framework.measure_latency(dummy_stt_model, framework.audio_data)
    print(f"STT Latency: {stt_latency:.4f} seconds")
    framework.evaluate_stt(predicted_stt)

    # Test audio splitting
    print("\nTesting audio splitting:")
    chunks = framework.split_audio_on_silence()
    print(f"Split into {len(chunks)} chunks.")
    if chunks:
        chunk_filename = framework.save_audio_chunk(chunks[0], "test/first_chunk.wav")
        print(f"First chunk saved to {chunk_filename}")
        framework.cleanup_temp_files(chunk_filename)

    # Clean up dummy files
    framework.cleanup_temp_files(dummy_audio_path, dummy_transcript_path, dummy_translation_path)
