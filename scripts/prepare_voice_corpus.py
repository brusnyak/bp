#!/usr/bin/env python3
"""
Normalize, dedupe, and (optionally) auto-transcribe a directory of raw voice
recordings into a clean speaker_voices.json corpus ready for
scripts/finetune_personal_voice.py.

Problem this solves (found for real in this repo's own speaker_voices/, 2026-08-21):
  - speaker_voices.json had duplicate entries pointing at the same file (id
    suffix "_dup2"), double-counting 4 of the then-5 real recordings in the
    training corpus that produced the current backend/tts/piper_models/
    en_US-personal-medium.onnx.
  - voice_rec_1m.wav and voice_rec_1m.m4a are the same take, once original and
    once as a format-converted copy - a naive "glob everything" ingest would
    count that take twice.
  - 15 of 24 entries pointed at speaker_voices/synth_*.wav files that don't
    exist on disk (the GPT-SoVITS bootstrap output from
    documentation/personal_voice_bootstrap_2026-08-19.md was never persisted
    into this directory) - finetune_personal_voice.py silently skips missing
    files, so this wasn't fatal, but the "24-entry manifest" the bootstrap doc
    describes only had ~9 entries that actually existed, over 5 unique files.

What this script does NOT do: transcription quality checking, audio
denoising/EQ, or the GPT-SoVITS synthetic-corpus bootstrap (separate,
offline-only step - see documentation/personal_voice_bootstrap_2026-08-19.md).

Usage:
    python scripts/prepare_voice_corpus.py --source-dir speaker_voices --write
    python scripts/prepare_voice_corpus.py --source-dir path/to/new/take --merge --write
"""
import argparse
import json
import subprocess
import sys
import wave
from pathlib import Path
from typing import Optional

import numpy as np

# Must run before any `backend.*` import below - scripts/ is not the project root, and
# scripts/temp_transcribe.py has the same fix but places it AFTER its backend imports
# (too late to matter). sys.path[0] is this file's directory when run as `python
# scripts/prepare_voice_corpus.py`, not the project root, so `backend` doesn't resolve
# without this.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

AUDIO_EXTENSIONS = {".wav", ".m4a", ".mp3", ".webm", ".ogg", ".flac", ".aac"}
DUPLICATE_SUFFIXES = ("_dup2", "_dup", "_copy", " (1)", "-copy")


def sanitize_stem(name: str) -> str:
    """Reject path separators / traversal in anything derived from a user-supplied filename
    before it's used to build an output path or passed to ffmpeg."""
    stem = Path(name).stem
    safe = "".join(c for c in stem if c.isalnum() or c in ("_", "-", " ")).strip()
    if not safe or safe in (".", ".."):
        raise ValueError(f"Unsafe or empty filename after sanitization: {name!r}")
    return safe


def normalized_stem(path: Path) -> str:
    """Strip extension and common copy/duplicate-suffix patterns, for duplicate-group hints."""
    stem = path.stem.lower()
    for suffix in DUPLICATE_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem.strip()


def ffmpeg_to_wav(src: Path, dst: Path, sample_rate: int) -> None:
    """Convert any input format to canonical mono 16-bit WAV. Arg-list subprocess call only -
    never shell=True, never string-interpolated - src/dst are Path objects, not raw user text."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
        "-ac", "1", "-ar", str(sample_rate), "-sample_fmt", "s16",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {src.name}: {result.stderr[-500:]}")


def audio_fingerprint(wav_path: Path, buckets: int = 20) -> np.ndarray:
    """Cheap content fingerprint: RMS energy across `buckets` equal-length segments of the
    normalized WAV. Robust to the same take being re-encoded through a different codec/
    extension (e.g. a .wav and its .m4a copy), without a real audio-fingerprinting dependency."""
    with wave.open(str(wav_path), "rb") as w:
        frames = w.readframes(w.getnframes())
    data = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
    if len(data) == 0:
        return np.zeros(buckets)
    chunks = np.array_split(data, buckets)
    return np.array([np.sqrt(np.mean(c ** 2)) if len(c) else 0.0 for c in chunks])


def wav_duration(wav_path: Path) -> float:
    with wave.open(str(wav_path)) as w:
        return w.getnframes() / w.getframerate()


def is_near_duplicate(fp_a: np.ndarray, fp_b: np.ndarray, dur_a: float, dur_b: float,
                       cosine_threshold: float = 0.985, duration_tolerance: float = 0.5) -> bool:
    if abs(dur_a - dur_b) > duration_tolerance:
        return False
    norm_a, norm_b = np.linalg.norm(fp_a), np.linalg.norm(fp_b)
    if norm_a == 0 or norm_b == 0:
        return norm_a == norm_b
    cosine = float(np.dot(fp_a, fp_b) / (norm_a * norm_b))
    return cosine >= cosine_threshold


def transcribe(wav_path: Path) -> Optional[str]:
    """Auto-transcribe a clip with no existing transcript, reusing the project's own STT
    stack (same pattern as scripts/temp_transcribe.py) instead of a new dependency."""
    try:
        from backend.stt.faster_whisper_stt import FasterWhisperSTT
        from backend.utils.audio_utils import load_audio
    except ImportError as e:
        print(f"  WARNING: STT unavailable ({e}), leaving transcript empty for {wav_path.name}", file=sys.stderr)
        return None
    audio_data, sample_rate = load_audio(str(wav_path), target_sr=16000)
    stt_model = transcribe._model if hasattr(transcribe, "_model") else FasterWhisperSTT(model_size="base", compute_type="int8")
    transcribe._model = stt_model  # cache across calls in one run
    segments, _, _ = stt_model.transcribe_audio(audio_data, sample_rate)
    text = " ".join(s.text for s in segments).strip()
    return text or None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-dir", default="speaker_voices")
    p.add_argument("--manifest", default=None, help="existing speaker_voices.json to reconcile against (default: <source-dir>/speaker_voices.json)")
    p.add_argument("--language", default="en")
    p.add_argument("--sample-rate", type=int, default=22050)
    p.add_argument("--auto-transcribe", action="store_true", help="run local STT on clips with no transcript")
    p.add_argument("--write", action="store_true", help="write the cleaned manifest back; default is dry-run/report-only")
    args = p.parse_args()

    source_dir = Path(args.source_dir).resolve()
    manifest_path = Path(args.manifest) if args.manifest else source_dir / "speaker_voices.json"

    old_entries = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    transcript_by_path = {}
    for e in old_entries:
        rel = e.get("path") or (f"{source_dir.name}/{e['filename']}" if "filename" in e else None)
        if rel and e.get("transcribed_text"):
            transcript_by_path.setdefault(Path(rel).name, e["transcribed_text"])

    missing = [e for e in old_entries if not (source_dir / Path(e.get("path", e.get("filename", ""))).name).exists()]
    if missing:
        print(f"Dropping {len(missing)} manifest entries with no file on disk:")
        for e in missing:
            print(f"  - {e.get('id')}: {e.get('path', e.get('filename'))}")

    raw_files = sorted(f for f in source_dir.iterdir() if f.suffix.lower() in AUDIO_EXTENSIONS)
    print(f"\nFound {len(raw_files)} raw audio file(s) in {source_dir}")

    work_dir = source_dir / ".corpus_work"
    work_dir.mkdir(exist_ok=True)

    normalized = []  # list of (safe_stem, original_path, normalized_wav_path, duration, fingerprint)
    for f in raw_files:
        try:
            safe_stem = sanitize_stem(f.name)
        except ValueError as e:
            print(f"  SKIP (unsafe filename): {e}")
            continue
        norm_wav = work_dir / f"{safe_stem}.norm.wav"
        ffmpeg_to_wav(f, norm_wav, args.sample_rate)
        dur = wav_duration(norm_wav)
        fp = audio_fingerprint(norm_wav)
        normalized.append({"stem": normalized_stem(f), "safe_stem": safe_stem, "orig": f, "norm": norm_wav, "dur": dur, "fp": fp})

    # Group near-duplicates (same normalized content, regardless of extension/filename suffix)
    groups: list[list[dict]] = []
    for item in normalized:
        placed = False
        for group in groups:
            if is_near_duplicate(item["fp"], group[0]["fp"], item["dur"], group[0]["dur"]):
                group.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])

    print(f"\n{len(normalized)} file(s) -> {len(groups)} unique take(s) after dedup:")
    canonical_entries = []
    for group in groups:
        # Prefer: has an existing transcript > original .wav (not a lossy re-encode) > longer filename (more descriptive)
        def rank(item):
            has_transcript = transcript_by_path.get(item["orig"].name) is not None
            is_wav = item["orig"].suffix.lower() == ".wav"
            return (has_transcript, is_wav, len(item["orig"].name))
        group.sort(key=rank, reverse=True)
        canonical = group[0]
        dupes = group[1:]
        if dupes:
            print(f"  '{canonical['orig'].name}' (kept) duplicates: {[d['orig'].name for d in dupes]}")
        else:
            print(f"  '{canonical['orig'].name}' (unique)")
        canonical_entries.append(canonical)

    # Transcripts: reuse existing, else auto-transcribe if requested
    new_manifest = []
    for item in canonical_entries:
        existing_text = transcript_by_path.get(item["orig"].name)
        transcript_source = "existing"
        text = existing_text
        if not text and args.auto_transcribe:
            print(f"  Transcribing {item['orig'].name} ...")
            text = transcribe(item["norm"])
            transcript_source = "auto-STT (faster-whisper base, not human-verified)"
        if not text:
            print(f"  WARNING: no transcript for {item['orig'].name} (run with --auto-transcribe, or add one manually) - excluded from training manifest")
            continue
        new_manifest.append({
            "id": item["safe_stem"],
            "name": item["safe_stem"],
            "language": args.language,
            "path": f"{source_dir.name}/{item['orig'].name}",
            "transcribed_text": text,
            "transcript_source": transcript_source,
        })

    total_seconds = sum(item["dur"] for item in canonical_entries if any(m["id"] == item["safe_stem"] for m in new_manifest))
    print(f"\nClean corpus: {len(new_manifest)} entries, {total_seconds:.1f}s ({total_seconds/60:.2f} min) of unique real audio.")

    if args.write:
        manifest_path.write_text(json.dumps(new_manifest, indent=2))
        print(f"Wrote {manifest_path}")
    else:
        print("(dry run - pass --write to save)")

    for f in work_dir.glob("*.norm.wav"):
        f.unlink()
    work_dir.rmdir()


if __name__ == "__main__":
    main()
