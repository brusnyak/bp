#!/usr/bin/env python3
"""Download a faster-whisper (CTranslate2) model into a plain directory.

    python scripts/fetch_whisper.py Systran/faster-whisper-base ct2_models/whisper-base

A plain directory avoids the Hugging Face cache, whose symlinks need Developer Mode or admin rights on Windows
(WinError 1314), and lets the app run offline. backend/stt/faster_whisper_stt.py prefers ct2_models/whisper-<size>.
"""
import sys

from huggingface_hub import snapshot_download

repo, out = sys.argv[1], sys.argv[2]
snapshot_download(repo, local_dir=out)
print("fetched", repo, "->", out)
