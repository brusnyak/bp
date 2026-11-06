"""Build the code package for Kaggle full-route kernel push.

Copies allowlisted repo files directly to /tmp/bp_full_route_pkg/ (no zips),
so Kaggle mounts them directly at /kaggle/input/bp-full-route-code-v*/.
"""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # -> BP repo root
OUT_DIR = Path("/tmp/bp_full_route_pkg")

# Allowlisted files/dirs to copy into the Kaggle package
ALLOWLIST = [
    "app.py",
    "backend/",
    "ui/",
    "languages/",
    "requirements.txt",
    "scripts/gpu_bench/kaggle_full_route/kaggle_full_route.py",
    "scripts/gpu_bench/kaggle_full_route/job.py",
]

EXCLUDE_PATTERNS = [
    ".venv*", ".git", "__pycache__", "*.pyc", ".env", "certs/",
    "speaker_voices/", "ct2_models/", "processed/", "node_modules/",
    "*.onnx", "*.bin", "*.pt", "*.pth", "*.m4a", "*.wav",
]


def _should_exclude(path: Path) -> bool:
    for pat in EXCLUDE_PATTERNS:
        import fnmatch
        if fnmatch.fnmatch(str(path.relative_to(REPO_ROOT)), pat) or fnmatch.fnmatch(path.name, pat):
            return True
    return False


def build():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    for entry in ALLOWLIST:
        src = REPO_ROOT / entry
        if not src.exists():
            print(f"WARNING: {src} does not exist, skipping")
            continue
        dst = OUT_DIR / entry
        if src.is_dir():
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*EXCLUDE_PATTERNS))
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    total = sum(1 for _ in OUT_DIR.rglob("*"))
    print(f"Package built at {OUT_DIR} ({total} entries)")


if __name__ == "__main__":
    build()