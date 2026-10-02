#!/usr/bin/env python3
"""Demo pre-flight: one command that says whether this machine can run the demo.

Checks the local repo assets a live demo needs (certs, voice models, MT models,
speaker-voice registry, proof clips, secrets file) and, with --server, the running
backend itself (HTTPS root + /api/voice-lab/status + engine list).

    python3 scripts/demo_preflight.py              # assets only, any python3
    python3 scripts/demo_preflight.py --server     # also probe https://localhost:8000

Exit code 0 = every required check passed (warnings allowed). Nothing is written.
See documentation/demo_runbook_2026-09-28.md for what to do about each failure.
"""

import argparse
import json
import os
import ssl
import sys
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_URL = "https://localhost:8000"

# Required to start the demo at all; missing = exit 1.
REQUIRED = [
    ("certs/cert.pem", "HTTPS/WSS certificate (else: make certs)"),
    ("certs/key.pem", "HTTPS/WSS key (else: make certs)"),
    (".env", "secrets file: GOOGLE_CLIENT_ID + JWT_SECRET (see documentation/amd_fetch_checklist.md)"),
    ("backend/tts/piper_models/sk_SK-personal-male-medium.onnx",
     "shipped personal SK voice (the demo voice)"),
    ("backend/tts/piper_models/sk_SK-personal-male-medium.onnx.json",
     "Piper sidecar config for the demo voice (without it TTS init fails and the WS never acks — hit on Linux 2026-09-28)"),
    ("backend/tts/piper_models/cs_CZ-jirka-medium.onnx", "Piper base for SK text"),
    ("ct2_models/Helsinki-NLP--opus-mt-en-sk/model.bin", "Opus-MT EN->SK"),
    ("ct2_models/Helsinki-NLP--opus-mt-sk-en/model.bin", "Opus-MT SK->EN"),
    ("speaker_voices/speaker_voices.json", "speaker voice registry"),
]

# Nice to have on stage; missing = warning only.
OPTIONAL = [
    ("processed/e2e_ensk_sk.wav", "fallback take 3: pre-rendered EN->SK proof clip"),
    ("ui/voice-lab/lab.html", "Voice Lab review page (fallback take 2)"),
    ("ui/live-speech/live.html", "live control room page"),
    ("models/openvoice_v2/checkpoints_v2/converter/config.json",
     "OpenVoice converter (only needed for the hybrid clone path, not for Piper SK)"),
    ("backend/tts/piper_models/en_US-personal-v2.onnx", "personal EN voice"),
]


# Owner-local assets: gitignored by design (personal voices cannot be re-downloaded). On a fresh clone they are expected to be
# missing and the app falls back to the generic Piper voice, so they are reported as warnings there, not failures.
OWNER_LOCAL = {
    "backend/tts/piper_models/sk_SK-personal-male-medium.onnx",
    "backend/tts/piper_models/sk_SK-personal-male-medium.onnx.json",
    "speaker_voices/speaker_voices.json",
}


def is_fresh_clone(repo_root):
    """No personal SK voice on disk = not the owner's machine."""
    return not os.path.exists(os.path.join(repo_root, "backend/tts/piper_models/sk_SK-personal-male-medium.onnx"))


def check_paths(entries, repo_root):
    results = []
    for rel, why in entries:
        path = os.path.join(repo_root, rel)
        results.append((os.path.exists(path), rel, why))
    return results


def check_env_secrets(repo_root):
    """Report which of the two demo-critical keys exist -- never their values."""
    path = os.path.join(repo_root, ".env")
    try:
        with open(path, encoding="utf-8") as f:
            keys = {line.split("=", 1)[0].strip() for line in f
                    if line.strip() and not line.strip().startswith("#") and "=" in line}
    except OSError:
        return []
    return [(key in keys, f".env:{key}", "needed for Google login / token signing")
            for key in ("GOOGLE_CLIENT_ID", "JWT_SECRET")]


def split_fresh(results):
    """On a fresh clone move owner-local misses (and GOOGLE_CLIENT_ID, only needed for login) out of the hard-required group."""
    hard, soft = [], []
    for ok, name, detail in results:
        local = name in OWNER_LOCAL or name == ".env:GOOGLE_CLIENT_ID"
        (soft if local and not ok else hard).append((ok, name, detail))
    return hard, soft


def check_server():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    results = []
    for label, url, expect in (
        ("GET / (home page)", f"{BASE_URL}/", "text/html"),
        ("GET /api/voice-lab/status", f"{BASE_URL}/api/voice-lab/status", None),
    ):
        try:
            with urllib.request.urlopen(url, context=ctx, timeout=10) as r:
                body = r.read().decode("utf-8", "replace")
                results.append((r.status == 200, label, f"HTTP {r.status}"))
                if label.endswith("status"):
                    info = json.loads(body)
                    engines = ", ".join(sorted(info.get("engines", {})))
                    results.append((bool(engines), "engines advertised", engines))
                    backends = info.get("hardware_backends", {})
                    results.append((bool(backends), "per-stage backends",
                                    json.dumps(backends)))
        except Exception as e:  # noqa: BLE001 - report anything as a failed check
            results.append((False, label, f"{type(e).__name__}: {e}"))
    return results


def report(title, results, miss="FAIL"):
    print(f"\n{title}")
    for ok, name, detail in results:
        print(f"  {'PASS' if ok else miss}  {name}  --  {detail}")
    return all(ok for ok, _, _ in results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", action="store_true",
                    help="also probe the running backend on https://localhost:8000")
    args = ap.parse_args()

    print(f"Demo pre-flight -- repo: {REPO_ROOT}")
    required = check_paths(REQUIRED, REPO_ROOT) + check_env_secrets(REPO_ROOT)
    soft = []
    if is_fresh_clone(REPO_ROOT):
        print("Fresh clone detected (no personal SK voice): owner-local assets are warnings; the generic Piper voice is used.")
        required, soft = split_fresh(required)
    required_ok = report("REQUIRED (demo cannot start without these)", required)
    if soft:
        report("OWNER-LOCAL (not in git; generic voice / no Google login without them)", soft, miss="WARN")
    optional_ok = report("OPTIONAL (fallbacks / non-default paths)",
                         check_paths(OPTIONAL, REPO_ROOT), miss="WARN")
    server_ok = True
    if args.server:
        server_ok = report("SERVER (make run must be up)", check_server())
    else:
        print("\nSERVER  skipped -- re-run with --server while `make run` is up")

    print("\n" + ("ALL REQUIRED CHECKS PASSED" if required_ok else "REQUIRED CHECKS FAILED")
          + ("" if optional_ok else "  (optional fallbacks missing -- demo still runs)"))
    return 0 if (required_ok and server_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
