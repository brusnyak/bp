#!/usr/bin/env python3
"""Refresh the live model catalog from Piper voices.json and Hugging Face.

Stdlib only (urllib). Writes cache/catalog_live.json (git-ignored).

    python scripts/catalog_refresh.py        # refresh once and exit
    python scripts/catalog_refresh.py --offline  # keep the curated file, skip network

The POST /api/catalog/refresh route in backend/main.py runs the same logic inline;
this script is the CLI equivalent for cron/manual refresh and for offline previews.

Never hides a model for licence reasons — licence is an informational label only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO_ROOT, "cache")
LIVE_CACHE_PATH = os.path.join(CACHE_DIR, "catalog_live.json")
CURATED_PATH = os.path.join(REPO_ROOT, "catalog", "curated.json")

USER_AGENT = "bp-router/1.0"
PIPER_VOICES_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/voices.json"
OPUS_MT_URL = ("https://huggingface.co/api/models?author=Helsinki-NLP&"
               "search=opus-mt&limit=100&full=false")
WHISPER_URL = ("https://huggingface.co/api/models?author=Systran&"
               "search=faster-whisper&limit=50&full=false")
HTTP_TIMEOUT = 60
MAX_OPUS_PAGES = 5


def _load_json_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _http_get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
        return json.load(resp), resp.headers


def _next_url_from_link(link_header):
    if not link_header:
        return None
    for part in link_header.split(","):
        part = part.strip()
        if 'rel="next"' in part:
            start = part.find("<")
            end = part.find(">", start + 1)
            if start != -1 and end != -1:
                return part[start + 1:end]
    return None


def _piper_voices():
    """Turn Piper voices.json into catalog model entries (stage tts)."""
    models = []
    resp, _ = _http_get_json(PIPER_VOICES_URL)
    voices = resp if isinstance(resp, list) else []
    for v in voices:
        key = v.get("key")
        lang = v.get("language") or {}
        code = lang.get("code")
        files = v.get("files") or {}
        onnx = None
        for path, finfo in files.items():
            if isinstance(finfo, dict) and path.endswith(".onnx"):
                onnx = finfo
                break
        if not key or not code or not onnx:
            continue
        size_bytes = onnx.get("size_bytes") or 0
        models.append({
            "id": f"piper-{key}",
            "stage": "tts",
            "name": f"Piper {key}",
            "engine": "piper",
            "languages": [code],
            "pair": None,
            "size_mb": round(size_bytes / 1_000_000, 1),
            "devices": ["cpu"],
            "min_ram_gb": 1,
            "min_vram_gb": None,
            "streaming": True,
            "clone": False,
            "installed": False,
            "fit": "good",
            "recommended": False,
            "why": "Piper voice from live voices.json.",
            "licence": {"label": "GPL-3.0-or-later (runtime)", "url": "https://huggingface.co/rhasspy/piper-voices"},
            "source": {
                "kind": "piper",
                "repo": "rhasspy/piper-voices",
                "files": {key: {"path": key + ".onnx",
                                "size_bytes": onnx.get("size_bytes"),
                                "md5_digest": onnx.get("md5_digest")}},
            },
            "notes": "Live from voices.json; installed detection at runtime.",
        })
    return models


def _opus_mt_pairs():
    """List Helsinki-NLP opus-mt pairs via HF API cursor pagination (stage mt)."""
    models = []
    url = OPUS_MT_URL
    seen = set()
    pages = 0
    while url and pages < MAX_OPUS_PAGES:
        resp, headers = _http_get_json(url)
        link = headers.get("Link", "")
        for item in resp:
            repo_id = item.get("id", "")
            if "opus-mt" not in repo_id or repo_id in seen:
                continue
            seen.add(repo_id)
            parts = repo_id.split("/")[-1]
            if parts.startswith("opus-mt-") and "-" in parts[len("opus-mt-"):]:
                rest = parts[len("opus-mt-"):]
                src, tgt = rest.split("-", 1)
                models.append({
                    "id": f"Helsinki-NLP--{repo_id.replace('/', '--')}",
                    "stage": "mt",
                    "name": f"Opus-MT {src}->{tgt}",
                    "engine": "opus-mt",
                    "languages": [src, tgt],
                    "pair": f"{src}-{tgt}",
                    "size_mb": 300,
                    "devices": ["cpu", "cuda"],
                    "min_ram_gb": 1,
                    "min_vram_gb": None,
                    "streaming": False,
                    "clone": False,
                    "installed": False,
                    "fit": "ok",
                    "recommended": False,
                    "why": f"Live HF entry {repo_id}.",
                    "licence": {"label": "verify per model card", "url": f"https://huggingface.co/{repo_id}"},
                    "source": {"kind": "hf", "repo": repo_id, "files": {}},
                    "notes": "Converted to CT2 at install time.",
                })
        url = _next_url_from_link(link)
        pages += 1
    return models


def _whisper_repos():
    """List Systran faster-whisper repos (stage stt)."""
    models = []
    resp, _ = _http_get_json(WHISPER_URL)
    for item in resp:
        repo_id = item.get("id", "")
        size = round((item.get("safetensors") or {}).get("total") or 0) / 1_000_000 or 140
        models.append({
            "id": f"Systran--{repo_id.replace('/', '--')}",
            "stage": "stt",
            "name": f"Whisper {repo_id.split('/')[-1]}",
            "engine": "faster_whisper",
            "languages": list(item.get("tags") or [])[:12] or ["en"],
            "pair": None,
            "size_mb": round(size, 1),
            "devices": ["cpu", "cuda"],
            "min_ram_gb": 2,
            "min_vram_gb": None,
            "streaming": False,
            "clone": False,
            "installed": False,
            "fit": "ok",
            "recommended": False,
            "why": f"Live HF entry {repo_id}.",
            "licence": {"label": "verify per model card", "url": f"https://huggingface.co/{repo_id}"},
            "source": {"kind": "hf", "repo": repo_id, "files": {}},
            "notes": "Downloads on first use via faster-whisper.",
        })
    return models


def refresh(offline=False):
    """Build the live payload: curated entries merged with live network lists."""
    curated = _load_json_file(CURATED_PATH)
    if isinstance(curated, dict) and isinstance(curated.get("models"), list):
        curated_models = curated["models"]
    elif isinstance(curated, list):
        curated_models = curated
    else:
        curated_models = []

    models = list(curated_models)
    errors = []

    if not offline:
        try:
            models.extend(_piper_voices())
        except Exception as exc:  # noqa: BLE001 — one source must not kill the refresh
            errors.append(f"piper: {type(exc).__name__}: {exc}")
        try:
            models.extend(_opus_mt_pairs())
        except Exception as exc:  # noqa: BLE001
            errors.append(f"opus-mt: {type(exc).__name__}: {exc}")
        try:
            models.extend(_whisper_repos())
        except Exception as exc:  # noqa: BLE001
            errors.append(f"whisper: {type(exc).__name__}: {exc}")

    payload = {
        "models": models,
        "errors": errors,
        "refreshed": time.time(),
        "source": {"curated": CURATED_PATH, "live": "network" if not offline else "offline"},
    }

    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = LIVE_CACHE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, LIVE_CACHE_PATH)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description="Refresh the live model catalog.")
    parser.add_argument("--offline", action="store_true",
                        help="skip network; write curated-only cache")
    args = parser.parse_args(argv)
    payload = refresh(offline=args.offline)
    print(json.dumps({
        "models": len(payload["models"]),
        "errors": payload["errors"],
        "cache": LIVE_CACHE_PATH,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
