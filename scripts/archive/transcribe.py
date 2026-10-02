#!/usr/bin/env python3
"""
Transcription script for screen recording videos.
Workflow: Video -> Audio -> Chunks -> Whisper Transcription -> Combined Transcript
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

VIDEO_FILE = "/Users/yegor/Documents/STU/BP/Screen Recording 2026-04-25 at 18.16.30.mov"
OUTPUT_DIR = "/Users/yegor/Documents/STU/BP/transcription_output"
AUDIO_FILE = os.path.join(OUTPUT_DIR, "audio.m4a")
CHUNK_DIR = os.path.join(OUTPUT_DIR, "chunks")
TRANSCRIPT_DIR = os.path.join(OUTPUT_DIR, "transcripts")


def run_cmd(cmd, desc=""):
    print(f"\n{'='*60}")
    if desc:
        print(f"{desc}")
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    print('='*60)
    result = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=False)
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        sys.exit(1)
    return result


def extract_audio():
    """Extract and compress audio from video using ffmpeg."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nExtracting audio from video...")
    print(f"Input: {VIDEO_FILE}")
    print(f"Output: {AUDIO_FILE}")
    
    cmd = [
        "ffmpeg", "-i", VIDEO_FILE,
        "-vn", "-acodec", "aac", "-b:a", "64k",
        "-y", AUDIO_FILE
    ]
    run_cmd(cmd, "Step 1: Extracting and compressing audio")
    print(f"\nAudio extracted successfully: {AUDIO_FILE}")


def split_audio(chunk_duration=1800):
    """Split audio into chunks (default 30 min = 1800 seconds)."""
    os.makedirs(CHUNK_DIR, exist_ok=True)
    print(f"\nSplitting audio into {chunk_duration//60}-minute chunks...")
    
    cmd = [
        "ffmpeg", "-i", AUDIO_FILE,
        "-f", "segment", "-segment_time", str(chunk_duration),
        "-c", "copy", "-reset_timestamps", "1",
        os.path.join(CHUNK_DIR, "part_%03d.m4a")
    ]
    run_cmd(cmd, "Step 2: Splitting audio into chunks")
    
    chunks = sorted(Path(CHUNK_DIR).glob("part_*.m4a"))
    print(f"\nCreated {len(chunks)} chunks:")
    for chunk in chunks:
        print(f"  - {chunk.name}")
    return list(chunks)


def transcribe_chunks(model="medium", language="en", chunks=None):
    """Transcribe audio chunks using Whisper."""
    import whisper
    import torch
    
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    
    if chunks is None:
        chunks = sorted(Path(CHUNK_DIR).glob("part_*.m4a"))
    
    # Use Metal GPU (MPS) if available, otherwise CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    print(f"\nLoading Whisper model: {model}")
    model = whisper.load_model(model, device=device)
    
    print(f"\nTranscribing {len(chunks)} chunk(s)...")
    transcripts = []
    
    for i, chunk_path in enumerate(chunks, 1):
        txt_path = os.path.join(TRANSCRIPT_DIR, f"{Path(chunk_path).stem}.txt")
        srt_path = os.path.join(TRANSCRIPT_DIR, f"{Path(chunk_path).stem}.srt")
        
        if os.path.exists(txt_path) and os.path.exists(srt_path):
            print(f"\n[{i}/{len(chunks)}] Skipping {os.path.basename(chunk_path)} (already transcribed)")
            with open(txt_path, "r", encoding="utf-8") as f:
                transcripts.append(f.read())
            continue
        
        print(f"\n[{i}/{len(chunks)}] Transcribing {os.path.basename(chunk_path)}...")
        result = model.transcribe(str(chunk_path), language=language, verbose=False)
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
        
        with open(srt_path, "w", encoding="utf-8") as f:
            for seg in result["segments"]:
                f.write(f"{seg['id']+1}\n")
                f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
                f.write(f"{seg['text'].strip()}\n\n")
        
        transcripts.append(result["text"])
        print(f"  Saved: {os.path.basename(txt_path)}, {os.path.basename(srt_path)}")
    
    return transcripts


def format_timestamp(seconds):
    """Format seconds to SRT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def combine_transcripts():
    """Combine all chunk transcripts into a single file."""
    txt_files = sorted(Path(TRANSCRIPT_DIR).glob("part_*.txt"))
    
    combined_txt = os.path.join(OUTPUT_DIR, "full_transcript.txt")
    combined_srt = os.path.join(OUTPUT_DIR, "full_transcript.srt")
    
    print(f"\nCombining {len(txt_files)} transcript(s)...")
    
    with open(combined_txt, "w", encoding="utf-8") as out:
        for txt_file in txt_files:
            with open(txt_file, "r", encoding="utf-8") as f:
                out.write(f.read().strip() + "\n\n")
    
    srt_files = sorted(Path(TRANSCRIPT_DIR).glob("part_*.srt"))
    idx = 1
    with open(combined_srt, "w", encoding="utf-8") as out:
        for srt_file in srt_files:
            with open(srt_file, "r", encoding="utf-8") as f:
                content = f.read()
                for block in content.strip().split("\n\n"):
                    lines = block.strip().split("\n")
                    if len(lines) >= 3:
                        out.write(f"{idx}\n")
                        out.write("\n".join(lines[1:]) + "\n\n")
                        idx += 1
    
    print(f"Saved combined transcript: {combined_txt}")
    print(f"Saved combined SRT: {combined_srt}")
    return combined_txt, combined_srt


def main():
    parser = argparse.ArgumentParser(description="Transcribe screen recording video")
    parser.add_argument("--model", default="medium", choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size (default: medium)")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    parser.add_argument("--chunk-duration", type=int, default=1800,
                        help="Chunk duration in seconds (default: 1800 = 30 min)")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio extraction")
    parser.add_argument("--skip-split", action="store_true", help="Skip audio splitting")
    parser.add_argument("--audio-only", action="store_true", help="Only extract audio, no transcription")
    args = parser.parse_args()
    
    print("="*60)
    print("SCREEN RECORDING TRANSCRIPTION SCRIPT")
    print("="*60)
    print(f"Video file: {VIDEO_FILE}")
    print(f"Model: {args.model}")
    print(f"Language: {args.language}")
    print(f"Chunk duration: {args.chunk_duration//60} minutes")
    print("="*60)
    
    if not os.path.exists(VIDEO_FILE):
        print(f"Error: Video file not found: {VIDEO_FILE}")
        sys.exit(1)
    
    # Step 1: Extract audio
    if not args.skip_audio:
        extract_audio()
    else:
        print("\nSkipping audio extraction...")
    
    if args.audio_only:
        print("\nAudio-only mode. Exiting.")
        return
    
    # Step 2: Split audio
    if not args.skip_split:
        chunks = split_audio(args.chunk_duration)
    else:
        print("\nSkipping audio split...")
        chunks = None
    
    # Step 3: Transcribe
    print(f"\n{'='*60}")
    print("Step 3: Transcribing with Whisper")
    print('='*60)
    transcribe_chunks(args.model, args.language, chunks)
    
    # Step 4: Combine
    print(f"\n{'='*60}")
    print("Step 4: Combining transcripts")
    print('='*60)
    combine_transcripts()
    
    print(f"\n{'='*60}")
    print("TRANSCRIPTION COMPLETE!")
    print('='*60)
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"  - Audio: {AUDIO_FILE}")
    print(f"  - Chunks: {CHUNK_DIR}")
    print(f"  - Individual transcripts: {TRANSCRIPT_DIR}")
    print(f"  - Full transcript: {os.path.join(OUTPUT_DIR, 'full_transcript.txt')}")
    print(f"  - Full SRT: {os.path.join(OUTPUT_DIR, 'full_transcript.srt')}")
    print("="*60)


if __name__ == "__main__":
    main()
