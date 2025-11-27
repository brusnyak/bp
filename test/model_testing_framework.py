import numpy as np
import soundfile as sf
import jiwer # For Word Error Rate calculation
import time # Import time
from typing import Callable, Tuple, List, Any, Dict, Optional # Import Dict
import time
import sacrebleu # Import sacrebleu for BLEU score
from nltk.translate.meteor_score import meteor_score # Import meteor_score
from nltk.tokenize import word_tokenize # Import word_tokenize for tokenization
import nltk # Import nltk for downloading punkt and wordnet

# Ensure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

from collections import deque

class ModelTestFramework:
    def __init__(self, audio_path: str, transcript_path: str, translation_path: Optional[str] = None, target_sr: int = 16000):
        self.audio_path = audio_path
        self.transcript_path = transcript_path
        self.translation_path = translation_path
        self.target_sr = target_sr
        self.audio_data, self.samplerate = self._load_and_resample_audio(audio_path, target_sr)
        self.ground_truth_transcript = self._load_transcript(transcript_path)
        self.ground_truth_translation = self._load_translation(translation_path) if translation_path else None

    def _load_and_resample_audio(self, audio_path: str, target_sr: int) -> Tuple[np.ndarray, int]:
        audio_data, original_samplerate = sf.read(audio_path, dtype='float32')
        if original_samplerate != target_sr:
            # Simple resampling, for more advanced scenarios, use librosa.resample
            from scipy.signal import resample
            num_samples = int(len(audio_data) * target_sr / original_samplerate)
            audio_data = resample(audio_data, num_samples)
            original_samplerate = target_sr # Update samplerate to target_sr after resampling
        return audio_data, original_samplerate

    def _load_transcript(self, transcript_path: str) -> str:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def _load_translation(self, translation_path: str) -> str:
        with open(translation_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def measure_latency(self, func: Callable, *args, **kwargs) -> Tuple[Any, float]:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        latency = end_time - start_time
        return result, latency

    def evaluate_stt(self, predicted_transcript: str) -> dict[str, float]:
        error = jiwer.wer(self.ground_truth_transcript, predicted_transcript)
        mer = jiwer.mer(self.ground_truth_transcript, predicted_transcript)
        wil = jiwer.wil(self.ground_truth_transcript, predicted_transcript)
        cer = jiwer.cer(self.ground_truth_transcript, predicted_transcript)
        return {"wer": error, "mer": mer, "wil": wil, "cer": cer}

    def evaluate_mt(self, predicted_translation: str) -> dict[str, float]:
        if self.ground_truth_translation is None:
            raise ValueError("Ground truth translation not loaded for MT evaluation.")
        
        # BLEU score
        bleu = sacrebleu.sentence_bleu(predicted_translation, [self.ground_truth_translation]).score

        # METEOR score
        # NLTK's meteor_score expects tokenized sentences
        reference_tokens = word_tokenize(self.ground_truth_translation)
        hypothesis_tokens = word_tokenize(predicted_translation)
        meteor = meteor_score([reference_tokens], hypothesis_tokens)

        return {"bleu": bleu, "meteor": meteor}
