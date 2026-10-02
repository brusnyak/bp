import re
import json
import os
import sys
import numpy as np # Import numpy for np.nan

def parse_report(report_content: str) -> dict:
    results = {
        "piper_tts": {},
        "f5_tts_en_en": [],
        "f5_tts_en_sk": [],
        "mt_accuracy": {},
        "speaker_similarity": {}
    }

    lines = report_content.splitlines()
    
    current_test_type = None
    current_f5_test_params = {}
    current_f5_test_metrics = {}
    temp_speaker_similarity = None

    for i, line in enumerate(lines):
        # Piper TTS section
        if "--- Running full pipeline test with STT model size: base and Piper TTS ---" in line:
            current_test_type = "piper_tts"
            current_f5_test_params = {}
            current_f5_test_metrics = {}
            temp_speaker_similarity = None
            continue
        
        # F5-TTS English-to-English section (initial non-parameterized)
        if "--- Running F5-TTS English-to-English voice cloning test (NFE: 4, Sway: -1, FixDur: None, CrossFade: 0.15) ---" in line:
            current_test_type = "f5_tts_en_en_initial"
            current_f5_test_params = {"nfe": 4.0, "cfg_strength": 2.0, "speed": 1.0}
            current_f5_test_metrics = {"speaker_similarity": np.nan} # Initialize speaker_similarity
            temp_speaker_similarity = None
            continue

        # F5-TTS English-to-English section (parameterized)
        f5_en_en_param_match = re.search(r"--- Running F5-TTS English-to-English voice cloning test \(NFE: ([\d.]+), CfgStrength: ([\d.]+), Speed: ([\d.]+)\) ---", line)
        if f5_en_en_param_match:
            current_test_type = "f5_tts_en_en_param"
            current_f5_test_params = {
                "nfe": float(f5_en_en_param_match.group(1)),
                "cfg_strength": float(f5_en_en_param_match.group(2)),
                "speed": float(f5_en_en_param_match.group(3))
            }
            current_f5_test_metrics = {"speaker_similarity": np.nan} # Initialize speaker_similarity
            temp_speaker_similarity = None
            continue

        # F5-TTS English-to-Slovak section (parameterized)
        f5_en_sk_param_match = re.search(r"--- Running F5-TTS English-to-Slovak translation test \(NFE: ([\d.]+), CfgStrength: ([\d.]+), Speed: ([\d.]+)\) ---", line)
        if f5_en_sk_param_match:
            current_test_type = "f5_tts_en_sk_param"
            current_f5_test_params = {
                "nfe": float(f5_en_sk_param_match.group(1)),
                "cfg_strength": float(f5_en_sk_param_match.group(2)),
                "speed": float(f5_en_sk_param_match.group(3))
            }
            current_f5_test_metrics = {"speaker_similarity": np.nan} # Initialize speaker_similarity
            temp_speaker_similarity = None
            continue

        # MT Accuracy Test section
        if "--- Running MT accuracy test for problematic phrase ---" in line:
            current_test_type = "mt_accuracy"
            current_f5_test_params = {}
            current_f5_test_metrics = {}
            temp_speaker_similarity = None
            continue

        # Speaker Similarity for Piper TTS section
        if "--- Calculating Speaker Similarity for Piper TTS ---" in line:
            current_test_type = "piper_similarity"
            current_f5_test_params = {}
            current_f5_test_metrics = {}
            temp_speaker_similarity = None
            continue

        # Extract metrics based on current_test_type
        if current_test_type:
            stt_latency_match = re.search(r"Average STT Latency: ([\d.]+)s", line)
            if stt_latency_match:
                current_f5_test_metrics["stt_latency"] = float(stt_latency_match.group(1))
                continue
            
            mt_latency_match = re.search(r"Average MT Latency: ([\d.]+)s", line)
            if mt_latency_match:
                current_f5_test_metrics["mt_latency"] = float(mt_latency_match.group(1))
                continue

            tts_latency_match = re.search(r"Average TTS Latency: ([\d.]+)s", line)
            if tts_latency_match:
                current_f5_test_metrics["tts_latency"] = float(tts_latency_match.group(1))
                continue

            total_e2e_latency_match = re.search(r"Total End-to-End Pipeline Latency: ([\d.]+)s", line)
            if total_e2e_latency_match:
                current_f5_test_metrics["total_e2e_latency"] = float(total_e2e_latency_match.group(1))
                continue
            
            wer_match = re.search(r"WER: ([\d.]+)", line)
            if wer_match:
                current_f5_test_metrics["wer"] = float(wer_match.group(1))
                continue

            mt_passed_match = re.search(r"Pipeline Test \(MT\): (PASSED|FAILED)", line)
            if mt_passed_match:
                current_f5_test_metrics["mt_passed"] = (mt_passed_match.group(1) == "PASSED")
                continue
            
            final_transcription_match = re.search(r"Final Transcription: '(.*?)'", line)
            if final_transcription_match:
                current_f5_test_metrics["final_transcription"] = final_transcription_match.group(1)
                continue

            final_translation_match = re.search(r"Final Translation: '(.*?)'", line)
            if final_translation_match:
                current_f5_test_metrics["final_translation"] = final_translation_match.group(1)
                continue

            # Speaker Similarity for F5-TTS (parameterized tests)
            f5_similarity_param_match = re.search(r"Speaker Similarity \(Original vs F5 E-(E|SK) - NFE: [\d.]+, CfgStrength: [\d.]+, Speed: [\d.]+\): ([\d.]+)", line)
            if f5_similarity_param_match:
                temp_speaker_similarity = float(f5_similarity_param_match.group(2))
                continue
            
            # Speaker Similarity for initial F5-TTS E-E test
            f5_initial_similarity_match = re.search(r"Speaker Similarity \(Original vs F5 E-E\): ([\d.]+)", line)
            if f5_initial_similarity_match and current_test_type == "f5_tts_en_en_initial":
                temp_speaker_similarity = float(f5_initial_similarity_match.group(1))
                continue

            # MT Accuracy specific
            mt_actual_translation_match = re.search(r"Actual Slovak \(Post-processed\): '(.*?)' \(MT Latency: ([\d.]+)s\)", line)
            if mt_actual_translation_match and current_test_type == "mt_accuracy":
                results["mt_accuracy"]["actual_translation"] = mt_actual_translation_match.group(1)
                results["mt_accuracy"]["mt_latency"] = float(mt_actual_translation_match.group(2))
                continue
            
            mt_accuracy_passed_match = re.search(r"MT Accuracy Test \(Problematic Phrase\): (PASSED|FAILED)", line)
            if mt_accuracy_passed_match and current_test_type == "mt_accuracy":
                results["mt_accuracy"]["passed"] = (mt_accuracy_passed_match.group(1) == "PASSED")
                continue

            # Piper Speaker Similarity specific
            piper_similarity_match = re.search(r"Speaker Similarity \(Original vs Piper Base E-SK\): ([\d.]+)", line)
            if piper_similarity_match and current_test_type == "piper_similarity":
                results["speaker_similarity"]["piper_base_e_sk"] = float(piper_similarity_match.group(1))
                continue

        # End of a test block
        if "--- Finished" in line:
            if current_test_type == "piper_tts":
                results["piper_tts"] = current_f5_test_metrics
            elif current_test_type in ["f5_tts_en_en_initial", "f5_tts_en_en_param"]:
                if temp_speaker_similarity is not None:
                    current_f5_test_metrics["speaker_similarity"] = temp_speaker_similarity
                results["f5_tts_en_en"].append({**current_f5_test_params, **current_f5_test_metrics})
            elif current_test_type == "f5_tts_en_sk_param":
                if temp_speaker_similarity is not None:
                    current_f5_test_metrics["speaker_similarity"] = temp_speaker_similarity
                results["f5_tts_en_sk"].append({**current_f5_test_params, **current_f5_test_metrics})
            
            current_test_type = None
            current_f5_test_params = {}
            current_f5_test_metrics = {}
            temp_speaker_similarity = None
            continue

    return results

if __name__ == "__main__":
    report_file_path = "report.txt"
    if not os.path.exists(report_file_path):
        print(f"Error: Report file not found at {report_file_path}")
        sys.exit(1)

    with open(report_file_path, "r", encoding="utf-8") as f:
        report_content = f.read()

    parsed_data = parse_report(report_content)
    
    # Convert np.nan to None for JSON serialization
    def convert_nan_to_none(obj):
        if isinstance(obj, dict):
            return {k: convert_nan_to_none(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_nan_to_none(elem) for elem in obj]
        elif isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    parsed_data_serializable = convert_nan_to_none(parsed_data)
    print(json.dumps(parsed_data_serializable, indent=2))

    output_json_path = "documentation/visuals/parsed_test_results.json"
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(parsed_data_serializable, f, indent=2)
    print(f"\nParsed results saved to {output_json_path}")
