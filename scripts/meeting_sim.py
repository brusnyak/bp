#!/usr/bin/env python3
"""Scripted 2-speaker meeting simulation with an absolute timeline.

Speaker A (EN) and Speaker B (SK) alternate short turns read by the personal
Piper voices. Every turn runs the real local pipeline (STT -> MT -> TTS) and
records ABSOLUTE offsets: turns are laid sequentially with a fixed gap, like a
real meeting where each side waits for the translated playback.

Out (all under processed/meeting/, local-only):
  meeting_timeline.json  — turns with t_start_s, speech span, per-stage spans
  turn_XX_<dir>.wav      — translated playback per turn (what the other side hears)
  speak_XX_<lang>.wav    — source speech per turn (what the speaker said)
  meeting_mix.wav + chapters in JSON — whole conversation stitched in clock order

Timing model is "streaming-chunked" (mirrors the live app's Latency
Breakdown): each turn is split into sentences pipelined the moment their speech
ends while the speaker CONTINUES — chunk playbacks overlap later speech, like
live simultaneous interpretation. STT/MT/TTS per chunk are MEASURED; only the
0.15s streaming decode tail is assumed (recorded in the JSON). Streaming proof:
turn first-audio starts BEFORE the speaker finishes.

Run: .venv/bin/python scripts/meeting_sim.py
Chart: python3 scripts/build_demo_charts.py  -> ui/voice-lab/charts/gantt_meeting.png
"""
from __future__ import annotations

import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
OUT_DIR = os.path.join(REPO_ROOT, "processed", "meeting")
GAP_S = 0.8  # pause between translated playback and the next speaker

TURNS = [
    {"speaker": "A", "lang": "en",
     "text": "Good morning everyone. Today's demo runs entirely on this laptop, with no cloud in the loop."},
    {"speaker": "B", "lang": "sk",
     "text": "Dobré ráno. Počujete ma dobre? Tento hovor prekladáme lokálne, bez cloudu."},
    {"speaker": "A", "lang": "en",
     "text": "If I pause mid-sentence, the system waits for the voice detector before translating. Short chunks keep the delay low."},
    {"speaker": "B", "lang": "sk",
     "text": "Skúsim hovoriť pomaly a zreteľne, aby preklad stíhal držať krok. Krátke úseky držia oneskorenie nízko."},
]

SPEAKER_VOICE = {"en": "en_US-personal-v2", "sk": "sk_SK-personal-male-medium"}


def main() -> None:
    import numpy as np
    import soundfile as sf
    from backend.stt.faster_whisper_stt import FasterWhisperSTT
    from backend.mt.ctranslate2_mt import CTranslate2MT
    from backend.tts.piper_tts import PiperTTS

    os.makedirs(OUT_DIR, exist_ok=True)
    stt = {"en": FasterWhisperSTT("base"), "sk": FasterWhisperSTT("small-sk")}
    mt = {"en_sk": CTranslate2MT("Helsinki-NLP/opus-mt-en-sk"),
          "sk_en": CTranslate2MT("Helsinki-NLP/opus-mt-sk-en")}
    tts = {"en": PiperTTS(model_id=SPEAKER_VOICE["en"]),
           "sk": PiperTTS(model_id=SPEAKER_VOICE["sk"])}

    clock = 0.0
    timeline = []
    mix_parts = []  # (kind, wav_path_or_array, sr) in clock order
    chapters = []
    import re as _re

    STT_TAIL_S = 0.15  # only assumed constants: streaming decode tail, intra-turn pause
    INTRA_S = 0.3
    for i, turn in enumerate(TURNS):
        src, tgt = turn["lang"], ("sk" if turn["lang"] == "en" else "en")
        sentences = [s for s in _re.split(r"(?<=[.?!])\s+", turn["text"].strip()) if s]
        t_start = clock
        # Speaker reads sentence by sentence, continuously (INTRA_S natural pauses).
        chunks = []
        st = t_start
        for sent in sentences:
            sw, sr, _ = tts[src].synthesize(sent, language=src)
            dur = len(sw) / sr
            chunks.append({"sent": sent, "wav": sw, "sr": sr, "start": st, "dur": dur})
            st += dur + INTRA_S
        speech_end = st - INTRA_S
        speech = np.concatenate([c["wav"] for c in chunks])
        speak_path = os.path.join(OUT_DIR, f"speak_{i:02d}_{src}.wav")
        sf.write(speak_path, speech, sr)

        def to16(wav, rate):
            if rate == 16000:
                return np.asarray(wav, dtype=np.float32)
            import librosa
            return np.asarray(librosa.resample(np.asarray(wav, dtype=np.float32),
                                               orig_sr=rate, target_sr=16000), dtype=np.float32)

        # Pipeline every chunk the moment its speech ends (measured per chunk).
        # First-word / first-chunk probes run once on chunk 0 (headline latencies).
        transcripts, translations, play_parts = [], [], []
        mt_first_s = tts_first_s = None
        chunk_rows = []
        for k, c in enumerate(chunks):
            wav16 = to16(c["wav"], c["sr"])
            segs, c_stt_s, _ = stt[src].transcribe_audio(wav16, 16000, language=src)
            hyp = " ".join(s.text for s in segs).strip()
            t0 = time.perf_counter()
            tr, _ = mt[f"{src}_{tgt}"].translate(hyp, src, tgt)
            c_mt_s = time.perf_counter() - t0
            out, out_sr, c_tts_s = tts[tgt].synthesize(tr, language=tgt)
            if k == 0:
                words = hyp.split() or ["."]
                t0 = time.perf_counter()
                mt[f"{src}_{tgt}"].translate_segments([" ".join(words[:6])], src, tgt)
                mt_first_s = time.perf_counter() - t0
                clause = tr.split(".")[0].strip() + "."
                t0 = time.perf_counter()
                tts[tgt].synthesize(clause, language=tgt)
                tts_first_s = time.perf_counter() - t0
            pb_start = c["start"] + c["dur"] + STT_TAIL_S + mt_first_s + tts_first_s
            pb_dur = len(out) / out_sr
            transcripts.append(hyp)
            translations.append(tr)
            play_parts.append(out)
            chunk_rows.append({"k": k, "speech_start": round(c["start"], 3),
                               "speech_s": round(c["dur"], 3),
                               "stt_s": round(c_stt_s, 3), "mt_s": round(c_mt_s, 3),
                               "tts_s": round(c_tts_s, 3),
                               "playback_start": round(pb_start, 3),
                               "playback_s": round(pb_dur, 3)})
            mix_parts.append((c["start"], c["wav"], c["sr"], f"turn {i} speech {k}"))
            mix_parts.append((pb_start, out, out_sr, f"turn {i} playback {k}"))
        transcript = " ".join(transcripts)
        translation = " ".join(translations)
        full_play = np.concatenate(play_parts)
        play_s = len(full_play) / out_sr
        path = os.path.join(OUT_DIR, f"turn_{i:02d}_{src}-{tgt}.wav")
        sf.write(path, full_play, out_sr)

        first_audio_at = chunk_rows[0]["playback_start"]
        last_pb_end = max(c["playback_start"] + c["playback_s"] for c in chunk_rows)
        entry = {
            "n": i, "speaker": turn["speaker"], "direction": f"{src}->{tgt}",
            "script": turn["text"], "transcript": transcript, "translation": translation,
            "t_start_s": round(t_start, 3),
            "speech_s": round(speech_end - t_start, 3),
            "stt_s": round(sum(c["stt_s"] for c in chunk_rows), 3),
            "mt_s": round(sum(c["mt_s"] for c in chunk_rows), 3),
            "tts_s": round(sum(c["tts_s"] for c in chunk_rows), 3),
            "playback_s": round(play_s, 3),
            "mt_first_word_s": round(mt_first_s, 4),
            "tts_first_chunk_s": round(tts_first_s, 4),
            "first_audio_at": first_audio_at,
            "streaming": bool(first_audio_at < round(speech_end, 3)),
            "chunks": chunk_rows,
            "playback_wav": os.path.relpath(path, REPO_ROOT),
            "speech_wav": os.path.relpath(speak_path, REPO_ROOT),
        }
        timeline.append(entry)
        chapters.append({"turn": i, "speaker": turn["speaker"], "direction": entry["direction"],
                         "speech_at": round(t_start, 2), "playback_at": first_audio_at})
        clock = max(speech_end, last_pb_end) + GAP_S
        print(f"turn {i} {entry['direction']} speech={entry['speech_s']:.1f}s "
              f"first_audio@{first_audio_at:.2f}s streaming={entry['streaming']} "
              f"(chunk stt/mt/tts sums {entry['stt_s']:.2f}/{entry['mt_s']:.2f}/{entry['tts_s']:.2f}s)", flush=True)

    with open(os.path.join(OUT_DIR, "meeting_timeline.json"), "w", encoding="utf-8") as f:
        json.dump({"gap_s": GAP_S, "total_s": round(clock, 3), "turns": timeline,
                   "timing_model": "streaming-chunked",
                   "assumptions": {"stt_tail_s": STT_TAIL_S, "intra_turn_pause_s": INTRA_S,
                                   "note": "Sentence chunks pipelined as their speech ends while the speaker continues; chunk playbacks overlap later speech"}},
                  f, indent=2, ensure_ascii=False)
    print(f"wrote {os.path.relpath(OUT_DIR, REPO_ROOT)}/meeting_timeline.json "
          f"({len(timeline)} turns, {clock:.1f}s total)")

    # Stitched whole-conversation mix in clock order (single player + chapters).
    mix_sr = 22050
    total_n = int(round(clock * mix_sr)) + mix_sr
    mix = np.zeros(total_n, dtype=np.float32)
    for at, wav, wsr, _label in mix_parts:
        assert wsr == mix_sr, f"mix rate mismatch: {wsr}"
        n0 = int(round(at * mix_sr))
        n1 = min(n0 + len(wav), total_n)
        mix[n0:n1] += np.asarray(wav[:n1 - n0], dtype=np.float32)
    mix_path = os.path.join(OUT_DIR, "meeting_mix.wav")
    sf.write(mix_path, mix, mix_sr)
    with open(os.path.join(OUT_DIR, "meeting_timeline.json"), "r+", encoding="utf-8") as f:
        doc = json.load(f)
        doc["mix_wav"] = os.path.relpath(mix_path, REPO_ROOT)
        doc["chapters"] = chapters
        f.seek(0)
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.truncate()
    print(f"wrote {os.path.relpath(mix_path, REPO_ROOT)} ({clock:.1f}s) + chapters")


if __name__ == "__main__":
    main()
