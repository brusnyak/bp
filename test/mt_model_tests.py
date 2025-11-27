import os
import pytest
import numpy as np

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__)))) # Add test directory to sys.path

from test.model_testing_framework import ModelTestFramework
from backend.mt.ctranslate2_mt import CTranslate2MT

@pytest.fixture(scope="module")
def mt_framework():
    audio_path = "test/My test speech_xtts_speaker_clean.wav"
    transcript_path = "test/My test speech transcript.txt"
    translation_path = "test/My test speech translation.txt"

    if not os.path.exists(audio_path):
        pytest.skip(f"Error: Audio file not found at {audio_path}")
    if not os.path.exists(transcript_path):
        pytest.skip(f"Error: Transcript file not found at {transcript_path}")
    if not os.path.exists(translation_path):
        pytest.skip(f"Error: Translation file not found at {translation_path}")

    return ModelTestFramework(
        audio_path=audio_path,
        transcript_path=transcript_path,
        translation_path=translation_path
    )

@pytest.fixture(scope="module")
def ctranslate2_mt_en_sk_model():
    return CTranslate2MT(model_path="Helsinki-NLP/opus-mt-en-sk", device="auto")

@pytest.fixture(scope="module")
def ctranslate2_mt_sk_en_model():
    return CTranslate2MT(model_path="Helsinki-NLP/opus-mt-sk-en", device="auto")

def test_ctranslate2_mt_en_sk(mt_framework: ModelTestFramework, ctranslate2_mt_en_sk_model: CTranslate2MT):
    print("\n--- Testing CTranslate2 MT (EN -> SK) ---")
    stt_output_text = mt_framework.ground_truth_transcript
    (translated_text_sk_tuple, latency_sk) = mt_framework.measure_latency(
        ctranslate2_mt_en_sk_model.translate, stt_output_text, src_lang="en", tgt_lang="sk"
    )
    translated_text_sk = translated_text_sk_tuple[0] # Extract the actual translated text
    print(f"CTranslate2MT Latency (EN->SK): {latency_sk:.4f} seconds")
    mt_results_sk = mt_framework.evaluate_mt(translated_text_sk)
    
    assert translated_text_sk is not None, "CTranslate2MT (EN->SK) translation failed."
    assert latency_sk > 0, "CTranslate2MT (EN->SK) latency not measured correctly."
    if mt_results_sk:
        print(f"CTranslate2MT BLEU (EN->SK): {mt_results_sk['bleu']:.4f}")
        print(f"CTranslate2MT METEOR (EN->SK): {mt_results_sk['meteor']:.4f}")
        assert mt_results_sk['bleu'] > 10, "CTranslate2MT (EN->SK) BLEU score is too low." # Example threshold

def test_ctranslate2_mt_sk_en(mt_framework: ModelTestFramework, ctranslate2_mt_sk_en_model: CTranslate2MT):
    print("\n--- Testing CTranslate2 MT (SK -> EN) ---")
    
    # For SK->EN translation, the input is the Slovak translation, and the reference is the English transcript.
    sk_input_text = mt_framework.ground_truth_translation
    en_reference_text_path = "test/My test speech transcript.txt" # This is the English ground truth

    # Create a new ModelTestFramework instance for SK->EN evaluation
    # This instance will have the English transcript as its ground_truth_translation for evaluation
    sk_en_eval_framework = ModelTestFramework(
        audio_path=mt_framework.audio_path, # Audio path is not directly used for MT evaluation, but required by constructor
        transcript_path=mt_framework.transcript_path, # Transcript path is not directly used for MT evaluation, but required by constructor
        translation_path=en_reference_text_path # This is the reference for SK->EN translation
    )

    (translated_text_en_tuple, latency_en) = sk_en_eval_framework.measure_latency(
        ctranslate2_mt_sk_en_model.translate, sk_input_text, src_lang="sk", tgt_lang="en"
    )
    translated_text_en = translated_text_en_tuple[0] # Extract the actual translated text
    print(f"CTranslate2MT Latency (SK->EN): {latency_en:.4f} seconds")
    mt_results_en = sk_en_eval_framework.evaluate_mt(translated_text_en) # Use the new framework for evaluation
    
    assert translated_text_en is not None, "CTranslate2MT (SK->EN) translation failed."
    assert latency_en > 0, "CTranslate2MT (SK->EN) latency not measured correctly."
    if mt_results_en:
        print(f"CTranslate2MT BLEU (SK->EN): {mt_results_en['bleu']:.4f}")
        print(f"CTranslate2MT METEOR (SK->EN): {mt_results_en['meteor']:.4f}")
        assert mt_results_en['bleu'] > 10, "CTranslate2MT (SK->EN) BLEU score is too low." # Example threshold
