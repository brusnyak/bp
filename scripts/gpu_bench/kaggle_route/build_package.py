#!/usr/bin/env python3
"""Build the private code tarball for the Kaggle route job from an allowlist (never a denylist).

  python3 scripts/gpu_bench/kaggle_route/build_package.py OUT_DIR

Writes OUT_DIR/bp-code.tar.gz, OUT_DIR/rainbow_en_90s.wav (owner's voice, 16 kHz mono, first 90 s)
and OUT_DIR/dataset-metadata.json (private). Refuses any member whose path hits a forbidden marker.
"""
import json, subprocess, sys, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ALLOW = ["backend", "ui", "app.py", "languages", "requirements.txt", "requirements-convert.txt",
         "requirements-dev.txt", "scripts/language_packs.py", "scripts/convert_models.py",
         "scripts/convert_whisper.py", "scripts/live/paced_replay.py"]
SKIP_SUFFIX = (".pyc", ".onnx", ".db", ".pem", ".key", ".wav", ".m4a", ".mp3", ".bin", ".pt", ".safetensors")
FORBIDDEN = ("/.env", "certs/", "sql_app.db", ".venv", "models/", "speaker_voices/", "processed/", "__pycache__")


def keep(name):
    n = "/" + name
    return not (name.endswith(SKIP_SUFFIX) or name.endswith(".onnx.json") or any(f in n for f in FORBIDDEN))


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tar_path = out / "bp-code.tar.gz"
    members = []
    with tarfile.open(tar_path, "w:gz") as t:
        for item in ALLOW:
            p = ROOT / item
            files = [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file())
            for f in files:
                rel = f.relative_to(ROOT).as_posix()
                if keep(rel):
                    t.add(f, arcname=rel)
                    members.append(rel)
    bad = [m for m in members if not keep(m)]
    assert not bad, bad
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-t", "90", "-i", str(ROOT / "speaker_voices/me_en_rainbow.m4a"),
                    "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out / "rainbow_en_90s.wav")], check=True)
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "BP route code 20261104", "id": "yegorby/bp-route-code-20261104",
        "licenses": [{"name": "other"}], "isPrivate": True}, indent=1))
    print(len(members), "files ->", tar_path, tar_path.stat().st_size, "bytes")


if __name__ == "__main__":
    main(sys.argv[1])
