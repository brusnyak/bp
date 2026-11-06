#!/usr/bin/env python3
"""Build the private code tarball for the Kaggle route job from an allowlist (never a denylist).

  python3 scripts/gpu_bench/kaggle_lang/build_package.py OUT_DIR

Writes OUT_DIR/bp-code.tar.gz, OUT_DIR/clip_en.wav (owner's voice, 16 kHz mono, first 90 s)
and OUT_DIR/dataset-metadata.json (private). Refuses any member whose path hits a forbidden marker.
"""
import json, shutil, subprocess, sys, tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ALLOW = ["backend", "ui", "app.py", "languages", "requirements.txt", "requirements-convert.txt",
         "requirements-dev.txt", "scripts/language_packs.py", "scripts/convert_models.py",
         "scripts/convert_whisper.py", "scripts/live/paced_replay.py", "scripts/live/summarize_replay.py", "scripts/live/fragment_quality.py"]
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
    for name, src in (("clip_en.wav", "me_en_rainbow.m4a"), ("clip_sk.wav", "me_sk_trhove.m4a"), ("clip_cs.wav", "me_cs_staromestske.m4a")):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-t", "60", "-i", str(ROOT / "speaker_voices" / src),
                        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out / name)], check=True)
    (out / "dataset-metadata.json").write_text(json.dumps({
        "title": "BP lang code 20261104", "id": "yegorby/bp-lang-code-20261104",
        "licenses": [{"name": "other"}], "isPrivate": True}, indent=1))
    for ext, v in [(e, v) for e in (".onnx", ".onnx.json") for v in ("sk_SK-lili-medium", "cs_CZ-jirka-medium", "en_US-john-medium")]:   # generic public Piper SK voice only; the personal fine-tune is never uploaded
        shutil.copy(ROOT / "backend/tts/piper_models" / (v + ext), out / (v + ext))
    print(len(members), "files ->", tar_path, tar_path.stat().st_size, "bytes")


if __name__ == "__main__":
    main(sys.argv[1])
