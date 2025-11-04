import os
import sys
import numpy as np

# Add the current working directory to sys.path for absolute imports
sys.path.append(os.getcwd())

from model_testing_framework import ModelTestFramework
from backend.mt.marian_mt import MarianMT
from backend.mt.ctranslate2_mt import CTranslate2MT # Import CTranslate2MT


def run_mt_tests():
    audio_path = "test/My test speech_xtts_speaker_clean.wav"
    transcript_path = "test/My test speech transcript.txt"
    translation_path = "test/My test speech translation.txt"

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return
    if not os.path.exists(transcript_path):
        print(f"Error: Transcript file not found at {transcript_path}")
        return
    if not os.path.exists(translation_path):
        print(f"Error: Translation file not found at {translation_path}")
        return

    framework = ModelTestFramework(
        audio_path=audio_path,
        transcript_path=transcript_path,
        translation_path=translation_path
    )

    stt_output_text = framework.reference_transcript

    # NLLB-200 is causing issues with language token handling and garbled output, so we are skipping it for now.
    # print("\n--- Testing NLLB-200 MT (EN -> SK) ---")
    # try:
    #     nllb_mt = NLLB_MT(model_id="facebook/nllb-200-distilled-600M", device="mps")
        
    #     translated_text_sk, latency_sk = framework.measure_latency(
    #         nllb_mt.translate, stt_output_text, src_lang="eng_Latn", tgt_lang="slk_Latn"
    #     )
    #     print(f"NLLB-200 Latency (EN->SK): {latency_sk:.4f} seconds")
    #     mt_results_sk = framework.evaluate_mt(translated_text_sk)
    #     if mt_results_sk:
    #         print(f"NLLB-200 BLEU (EN->SK): {mt_results_sk['bleu']:.4f}")
    #         print(f"NLLB-200 METEOR (EN->SK): {mt_results_sk['meteor']:.4f}")

    # except Exception as e:
    #     print(f"Error testing NLLB-200 (EN -> SK): {e}")
    #     import traceback
    #     traceback.print_exc()

    print("\n--- Testing MarianMT (EN -> SK) ---")
    try:
        marian_mt_en_sk = MarianMT(model_id="Helsinki-NLP/opus-mt-en-sk", device="mps")
        translated_text_sk, latency_sk = framework.measure_latency(
            marian_mt_en_sk.translate, stt_output_text, src_lang="en", tgt_lang="sk"
        )
        print(f"MarianMT Latency (EN->SK): {latency_sk:.4f} seconds")
        mt_results_sk = framework.evaluate_mt(translated_text_sk)
        if mt_results_sk:
            print(f"MarianMT BLEU (EN->SK): {mt_results_sk['bleu']:.4f}")
            print(f"MarianMT METEOR (EN->SK): {mt_results_sk['meteor']:.4f}")

    except Exception as e:
        print(f"Error testing MarianMT (EN -> SK): {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing MarianMT (SK -> EN) ---")
    try:
        sk_input_text = framework.reference_translation
        en_reference_text = framework.reference_transcript

        marian_mt_sk_en = MarianMT(model_id="Helsinki-NLP/opus-mt-sk-en", device="mps")
        translated_text_en, latency_en = framework.measure_latency(
            marian_mt_sk_en.translate, sk_input_text, src_lang="sk", tgt_lang="en"
        )
        print(f"MarianMT Latency (SK->EN): {latency_en:.4f} seconds")
        
        original_reference_translation = framework.reference_translation
        framework.reference_translation = en_reference_text
        mt_results_en = framework.evaluate_mt(translated_text_en)
        if mt_results_en:
            print(f"MarianMT BLEU (SK->EN): {mt_results_en['bleu']:.4f}")
            print(f"MarianMT METEOR (SK->EN): {mt_results_en['meteor']:.4f}")
        framework.reference_translation = original_reference_translation

    except Exception as e:
        print(f"Error testing MarianMT (SK -> EN): {e}")
        import traceback
        traceback.print_exc()

    # SeamlessM4T-v2 is causing segmentation faults on MPS, so we are skipping it for now.
    # print("\n--- Testing SeamlessM4T-v2 MT (EN -> SK) ---")
    # try:
    #     seamless_m4t_mt = SeamlessM4T_MT(model_id="facebook/seamless-m4t-v2-large", device="mps")
    #     translated_text_sk, latency_sk = framework.measure_latency(
    #         seamless_m4t_mt.translate, stt_output_text, src_lang="eng", tgt_lang="slk"
    #     )
    #     print(f"SeamlessM4T-v2 Latency (EN->SK): {latency_sk:.4f} seconds")
    #     mt_results_sk = framework.evaluate_mt(translated_text_sk)
    #     if mt_results_sk:
    #         print(f"SeamlessM4T-v2 BLEU (EN->SK): {mt_results_sk['bleu']:.4f}")
    #         print(f"SeamlessM4T-v2 METEOR (EN->SK): {mt_results_sk['meteor']:.4f}")

    # except Exception as e:
    #     print(f"Error testing SeamlessM4T-v2 (EN -> SK): {e}")
    #     import traceback
    #     traceback.print_exc()

    # print("\n--- Testing SeamlessM4T-v2 MT (SK -> EN) ---")
    # try:
    #     sk_input_text = framework.reference_translation
    #     en_reference_text = framework.reference_transcript

    #     seamless_m4t_mt_sk_en = SeamlessM4T_MT(model_id="facebook/seamless-m4t-v2-large", device="mps")
    #     translated_text_en, latency_en = framework.measure_latency(
    #         seamless_m4t_mt_sk_en.translate, sk_input_text, src_lang="slk", tgt_lang="eng"
    #     )
    #     print(f"SeamlessM4T-v2 Latency (SK->EN): {latency_en:.4f} seconds")
        
    #     original_reference_translation = framework.reference_translation
    #     framework.reference_translation = en_reference_text
    #     mt_results_en = framework.evaluate_mt(translated_text_en)
    #     if mt_results_en:
    #         print(f"SeamlessM4T-v2 BLEU (SK->EN): {mt_results_en['bleu']:.4f}")
    #         print(f"SeamlessM4T-v2 METEOR (SK->EN): {mt_results_en['meteor']:.4f}")
    #     framework.reference_translation = original_reference_translation

    # except Exception as e:
    #     print(f"Error testing SeamlessM4T-v2 (SK -> EN): {e}")
    #     import traceback
    #     traceback.print_exc()

    print("\n--- Testing CTranslate2 MT (EN -> SK) ---")
    try:
        ctranslate2_mt_en_sk = CTranslate2MT(model_path="Helsinki-NLP/opus-mt-en-sk", device="auto")
        translated_text_sk, latency_sk = framework.measure_latency(
            ctranslate2_mt_en_sk.translate, stt_output_text, src_lang="en", tgt_lang="sk"
        )
        print(f"CTranslate2MT Latency (EN->SK): {latency_sk:.4f} seconds")
        mt_results_sk = framework.evaluate_mt(translated_text_sk)
        if mt_results_sk:
            print(f"CTranslate2MT BLEU (EN->SK): {mt_results_sk['bleu']:.4f}")
            print(f"CTranslate2MT METEOR (EN->SK): {mt_results_sk['meteor']:.4f}")

    except Exception as e:
        print(f"Error testing CTranslate2MT (EN -> SK): {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Testing CTranslate2 MT (SK -> EN) ---")
    try:
        sk_input_text = framework.reference_translation
        en_reference_text = framework.reference_transcript

        # CTranslate2 requires a separate converted model for SK->EN
        # Assuming "Helsinki-NLP/opus-mt-sk-en" is also converted and available
        ctranslate2_mt_sk_en = CTranslate2MT(model_path="Helsinki-NLP/opus-mt-sk-en", device="auto")
        translated_text_en, latency_en = framework.measure_latency(
            ctranslate2_mt_sk_en.translate, sk_input_text, src_lang="sk", tgt_lang="en"
        )
        print(f"CTranslate2MT Latency (SK->EN): {latency_en:.4f} seconds")
        
        original_reference_translation = framework.reference_translation
        framework.reference_translation = en_reference_text
        mt_results_en = framework.evaluate_mt(translated_text_en)
        if mt_results_en:
            print(f"CTranslate2MT BLEU (SK->EN): {mt_results_en['bleu']:.4f}")
            print(f"CTranslate2MT METEOR (SK->EN): {mt_results_en['meteor']:.4f}")
        framework.reference_translation = original_reference_translation

    except Exception as e:
        print(f"Error testing CTranslate2MT (SK -> EN): {e}")
        import traceback
        traceback.print_exc()

    # T5 is performing very poorly, so we are skipping it.
    # print("\n--- Testing T5 MT (EN -> SK) ---")
    # try:
    #     t5_mt_en_sk = T5_MT(model_id="t5-small", device="mps")
    #     translated_text_sk, latency_sk = framework.measure_latency(
    #         t5_mt_en_sk.translate, stt_output_text, src_lang="en", tgt_lang="sk"
    #     )
    #     print(f"T5 MT Latency (EN->SK): {latency_sk:.4f} seconds")
    #     mt_results_sk = framework.evaluate_mt(translated_text_sk)
    #     if mt_results_sk:
    #         print(f"T5 MT BLEU (EN->SK): {mt_results_sk['bleu']:.4f}")
    #         print(f"T5 MT METEOR (EN->SK): {mt_results_sk['meteor']:.4f}")

    # except Exception as e:
    #     print(f"Error testing T5 MT (EN -> SK): {e}")
    #     import traceback
    #     traceback.print_exc()

    # print("\n--- Testing T5 MT (SK -> EN) ---")
    # try:
    #     sk_input_text = framework.reference_translation
    #     en_reference_text = framework.reference_transcript

    #     t5_mt_sk_en = T5_MT(model_id="t5-small", device="mps")
    #     translated_text_en, latency_en = framework.measure_latency(
    #         t5_mt_sk_en.translate, sk_input_text, src_lang="sk", tgt_lang="en"
    #     )
    #     print(f"T5 MT Latency (SK->EN): {latency_en:.4f} seconds")
        
    #     original_reference_translation = framework.reference_translation
    #     framework.reference_translation = en_reference_text
    #     mt_results_en = framework.evaluate_mt(translated_text_en)
    #     if mt_results_en:
    #         print(f"T5 MT BLEU (SK->EN): {mt_results_en['bleu']:.4f}")
    #         print(f"T5 MT METEOR (SK->EN): {mt_results_en['meteor']:.4f}")
    #     framework.reference_translation = original_reference_translation

    # except Exception as e:
    #     print(f"Error testing T5 MT (SK -> EN): {e}")
    #     import traceback
    #     traceback.print_exc()


if __name__ == "__main__":
    run_mt_tests()
