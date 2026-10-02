#!/usr/bin/env python3
"""
Generate synthetic test audio for male_alldata_denoised.wav to benchmark STT.
"""

import sys, os, json
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Use the same TTS setup as engine_ab.py but with a different model ID if needed
from backend.tts.piper_tts import PiperTTS
import soundfile as sf

# The same text from engine_ab.py
TEXT = (
    "Včera ráno som išiel na trh kúpiť čerstvý chlieb a mlieko. "
    "Stretol som tam starého priateľa Ľuba, ktorý predával med a syry. "
    "Porozprával som mu o svojej práci a o dlhej ceste vlakom cez hory a doliny."
)

def synth_piper(model_id, name):
    """Synthesize TEXT using the given Piper model ID."""
    from backend.tts.piper_tts import PiperTTS
    import soundfile as sf
    tts = PiperTTS(model_id=model_id)
    wav, sr, syn_s = tts.synthesize(TEXT, language="sk")
    path = os.path.join(REPO_ROOT, "processed", "engine_ab", name + ".wav")
    sf.write(path, wav, sr)
    audio_s = len(wav) / sr
    return {"engine": name, "syn_s": round(syn_s, 3), "audio_s": round(audio_s, 2),
            "rtf": round(syn_s / audio_s, 3), "wav": path}

def main():
    # 1. Create synthetic male_alldata_denoised.wav from the shipped voice (sk_SK-personal-male-medium)
    # Note: this is synthetic, not the real denoised recording, but it's the same phonetic content
    print("Generating synthetic male_alldata_denoised.wav from shipped voice...")
    synth = synth_piper("sk_SK-personal-male-medium", "male_alldata_denoised")
    print(f"  Done: {synth['wav']}, RTF={synth['rtf']}, len={synth['audio_s']}s")

    # 2. Use the existing piper_omni_hq.wav (from the synthetic OmniVoice test)
    omni_path = os.path.join(REPO_ROOT, "processed", "engine_ab", "piper_omni_hq.wav")
    if not os.path.exists(omni_path):
        print("  piper_omni_hq.wav not found, synthesizing it...")
        synth_omni = synth_piper("me_omni_piper_sk", "piper_omni_hq")
    else:
        import soundfile as sf
        info = sf.info(omni_path)
        audio_s = info.duration
        synth_omni = {"engine": "piper_omni_hq", "wav": omni_path, "audio_s": round(audio_s, 2), "rtf": None}
        print(f"  Using existing: {omni_path}, len={audio_s}s")

    # 3. Compare the two using the same STT as engine_ab.py (small-sk)
    print()
    print("Comparing STT on synthetic male_alldata_denoised.wav vs piper_omni_hq.wav")
    print("Text:", TEXT)
    print()
    
    import librosa, numpy as np, jiwer, re, time
    from backend.stt.faster_whisper_stt import FasterWhisperSTT
    
    def norm(t):
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower(), flags=re.UNICODE)).strip()
    
    ref = norm(TEXT)
    
    # Run both through STT small-sk
    stt_sk = FasterWhisperSTT(model_size="small")
    
    for name, wav_path in [("male_alldata_denoised", synth["wav"]), ("piper_omni_hq", omni_path)]:
        wav, sr = librosa.load(wav_path, sr=16000, mono=True)
        start = time.perf_counter()
        segs, stt_s, _ = stt_sk.transcribe_audio(np.asarray(wav, dtype=np.float32), sr, language="sk")
        wall = time.perf_counter() - start
        hyp = norm(" ".join(x.text for x in segs))
        wer = jiwer.wer(ref, hyp)
        cer = jiwer.cer(ref, hyp)
        audio_s = len(wav) / sr
        rtf = stt_s / audio_s
        
        print(f"{name}:")
        print(f"  RTF = {rtf:.3f}, STT = {stt_s:.3f}s, wall = {wall:.3f}s")
        print(f"  WER = {wer:.4f}, CER = {cer:.4f}")
        print(f"  Hyp: {hyp[:120]}...")
        print()

    # 4. Write a summary file for documentation
    summary = {
        "text": TEXT,
        "synthetic_male_alldata_denoised_wav": synth["wav"],
        "piper_omni_hq_wav": omni_path,
        "benchmark": {
            "male_alldata_denoised": {
                "rtf": None, "wer": None, "cer": None, "stt_s": None, "wall_s": None
            },
            "piper_omni_hq": {
                "rtf": None, "wer": None, "cer": None, "stt_s": None, "wall_s": None
            }
        }
    }
    
    # Compute final results
    for name, wav_path in [("male_alldata_denoised", synth["wav"]), ("piper_omni_hq", omni_path)]:
        wav, sr = librosa.load(wav_path, sr=16000, mono=True)
        start = time.perf_counter()
        segs, stt_s, _ = stt_sk.transcribe_audio(np.asarray(wav, dtype=np.float32), sr, language="sk")
        wall = time.perf_counter() - start
        hyp = norm(" ".join(x.text for x in segs))
        wer = jiwer.wer(ref, hyp)
        cer = jiwer.cer(ref, hyp)
        audio_s = len(wav) / sr
        rtf = stt_s / audio_s
        
        summary["benchmark"][name] = {
            "rtf": round(rtf, 4),
            "stt_s": round(stt_s, 3),
            "wall_s": round(wall, 3),
            "wer": round(wer, 4),
            "cer": round(cer, 4),
            "hyp": hyp
        }
    
    out_path = os.path.join(REPO_ROOT, "processed", "male_alldata_denoised_benchmark.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote comparison summary to {out_path}")

if __name__ == "__main__":
    main()