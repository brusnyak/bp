import pytest
import numpy as np
import os
import soundfile as sf
import time # Import time for latency measurement
import sys # Import sys for standalone execution
from backend.tts.coqui_tts import CoquiTTS

# Define a dummy speaker WAV file for testing
DUMMY_SPEAKER_WAV = "test_speaker_coqui.wav"
DUMMY_SPEAKER_SR = 16000 # Coqui XTTS v2 expects 16kHz for speaker_wav
DUMMY_SPEAKER_DURATION = 5 # seconds

# Output directory for test audio
OUTPUT_DIR = "test_output/coqui_tts_tests"

def setup_dummy_speaker_wav():
    """Creates a dummy speaker WAV file for tests."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(DUMMY_SPEAKER_WAV):
        dummy_audio = np.random.rand(DUMMY_SPEAKER_SR * DUMMY_SPEAKER_DURATION).astype(np.float32)
        sf.write(DUMMY_SPEAKER_WAV, dummy_audio, DUMMY_SPEAKER_SR)
    return DUMMY_SPEAKER_WAV

def cleanup_dummy_speaker_wav():
    """Cleans up the dummy speaker WAV file."""
    if os.path.exists(DUMMY_SPEAKER_WAV):
        os.remove(DUMMY_SPEAKER_WAV)

# Fixture for pytest, also callable directly for standalone execution
@pytest.fixture(scope="module", autouse=True)
def coqui_tts_setup_teardown():
    """Sets up and tears down resources for CoquiTTS tests."""
    setup_dummy_speaker_wav()
    yield
    cleanup_dummy_speaker_wav()

@pytest.fixture(scope="module")
def coqui_tts_model():
    """Initializes and returns a CoquiTTS model instance."""
    try:
        model = CoquiTTS(device="cpu") # Force CPU for M1 Pro as per notes
        return model
    except Exception as e:
        pytest.fail(f"Failed to initialize CoquiTTS model: {e}")

def test_coqui_tts_initialization(coqui_tts_model):
    """Tests if the CoquiTTS model initializes correctly."""
    assert coqui_tts_model.model is not None, "CoquiTTS model should be initialized."
    assert coqui_tts_model.device == "cpu", "CoquiTTS should be forced to CPU on M1 Pro."
    assert coqui_tts_model.sample_rate == 24000, "CoquiTTS XTTS v2 sample rate should be 24000 Hz."

@pytest.mark.parametrize("text, language", [
    ("Hello, this is a test of my voice for cloning purposes.", "en"),
    ("Dobrý den, toto je test mého hlasu pro účely klonování.", "cs"),
])
def test_coqui_tts_synthesis(coqui_tts_model, text, language):
    """Tests speech synthesis with voice cloning for different languages."""
    speaker_wav_path = DUMMY_SPEAKER_WAV # Use the dummy speaker WAV
    audio_waveform, sample_rate, tts_time = coqui_tts_model.synthesize(
        text=text,
        language=language,
        speaker_wav_path=speaker_wav_path
    )

    assert audio_waveform is not None, f"Synthesis failed for language {language}."
    assert isinstance(audio_waveform, np.ndarray), "Synthesized audio should be a numpy array."
    assert audio_waveform.size > 0, "Synthesized audio waveform should not be empty."
    assert sample_rate == coqui_tts_model.sample_rate, f"Sample rate mismatch for {language}."
    assert tts_time > 0, "Synthesis time should be greater than 0."

    # Save the output for inspection and potential comparison
    output_filename = os.path.join(OUTPUT_DIR, f"coqui_output_{language}.wav")
    sf.write(output_filename, audio_waveform, sample_rate)
    print(f"Saved test audio to {output_filename}")
    print(f"CoquiTTS Synthesis Latency for {language}: {tts_time:.4f}s")

def test_coqui_tts_synthesis_no_speaker_wav(coqui_tts_model):
    """Tests synthesis failure when speaker WAV path is invalid."""
    text = "This should fail."
    language = "en"
    invalid_speaker_wav = "non_existent_speaker.wav"

    audio_waveform, sample_rate, tts_time = coqui_tts_model.synthesize(
        text=text,
        language=language,
        speaker_wav_path=invalid_speaker_wav
    )

    assert audio_waveform is None, "Synthesis should fail with invalid speaker WAV."
    assert sample_rate is None, "Sample rate should be None with invalid speaker WAV."
    assert tts_time == 0.0, "Synthesis time should be 0.0 with invalid speaker WAV."

# Standalone execution logic
if __name__ == "__main__":
    print("Running CoquiTTS standalone test...")
    
    # Ensure output directory exists for standalone execution
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Setup
    speaker_wav = setup_dummy_speaker_wav()
    
    try:
        coqui_tts = CoquiTTS(device="cpu")
        
        test_cases = [
            # Original test cases with simplified filenames
            ("Hello, this is a test of my voice for cloning purposes.", "en", "hello_dummy.wav"),
            ("Dobrý den, toto je test mého hlasu pro účely klonování.", "cs", "dobry_den_dummy.wav"),

            # Test cases with specific audio files for voice cloning duration measurement
            # en->en with test/Hello.wav as speaker reference
            ("Hello, this is a test of my voice for cloning purposes.", "en", "hello_cloned_en_en.wav", "test/Hello.wav"),
            # en->cs with test/My test speech_xtts_speaker_clean.wav as speaker reference
            ("In this experiment, the system converts spoken English into text, translates it into Czech and then synthesizes it back into speech.", "cs", "speech_cloned_en_cs.wav", "test/My test speech_xtts_speaker_clean.wav"),
        ]

        for i, (text, language, output_filename_suffix, *speaker_ref_path) in enumerate(test_cases):
            print(f"\n--- Test Case {i+1}: Language: {language}, Text: '{text[:50]}...' ---")
            
            current_speaker_wav = speaker_ref_path[0] if speaker_ref_path else speaker_wav
            
            start_time_e2e = time.perf_counter()
            audio_waveform, sample_rate, tts_time = coqui_tts.synthesize(
                text=text,
                language=language,
                speaker_wav_path=current_speaker_wav
            )
            end_time_e2e = time.perf_counter()

            if audio_waveform is not None:
                output_path = os.path.join(OUTPUT_DIR, f"coqui_output_{output_filename_suffix}")
                try:
                    sf.write(output_path, audio_waveform, sample_rate)
                    print(f"  Saved test audio to {output_path}")
                    print(f"  CoquiTTS Synthesis Latency (TTS only): {tts_time:.4f}s")
                    print(f"  CoquiTTS End-to-End Latency (including model load if first time): {end_time_e2e - start_time_e2e:.4f}s")
                except Exception as write_error:
                    print(f"  ERROR: Failed to write audio to {output_path}: {write_error}", file=sys.stderr)
            else:
                print(f"  Synthesis failed for text: '{text[:50]}...'")
    except Exception as e:
        print(f"An error occurred during standalone test: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # Teardown
        cleanup_dummy_speaker_wav()
    
    print("\nCoquiTTS standalone test finished.")
