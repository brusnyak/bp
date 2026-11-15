#!/usr/bin/env python3
"""Check tracked text for private project framing; no dependencies or writes."""

from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = ("documentation/evidence/", "bench/results/", "ui/vendor/")
# Adjacent literals keep the rule definitions from matching their own source.
PATTERN = re.compile(
    "(?<![a-z])the" "sis|bache" "lor|\\bS" "TU\\b|\\bF" "EI\\b|super" "visor|"
    "hand" "ler(?!s\\b)|picky old " "man|Miná" "rik|Mina" "rik|\\bL" "HC\\b|"
    "owner " "decision|the " "owner",
    re.IGNORECASE,
)

# Exact technical APIs, callback variables and connection lifecycle messages.
CALLBACKS = re.compile(
    r"\b(?:_rate_limit_exceeded_hand" r"ler|add_exception_hand" r"ler|"
    r"RotatingFileHand" r"ler|addHand" r"ler)\b|"
    r"\b(?:WebSocket|event) hand" r"ler\b|\bhand" r"ler finished\b|"
    r"\bhand" r"ler(?=\s*(?:=|\(|\)|,|\.))",
    re.IGNORECASE,
)


def main():
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    hits = 0
    for name in paths:
        if not name or name.startswith(EXCLUDED):
            continue
        path = ROOT / name
        # Symlinks can point at private assets outside the tracked tree.
        if path.is_symlink() or not path.is_file():
            continue
        raw = path.read_bytes()
        if b"\0" in raw:
            continue
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(content.splitlines(), 1):
            # Connection callbacks and logging APIs use this word technically.
            # Keep those code concepts intact; documentation gets no exemption.
            matches = [match for match in PATTERN.finditer(line)
                       if not (path.suffix in {".py", ".js", ".cjs", ".mjs"}
                               and match.group().lower() == "hand" "ler"
                               and any(start <= match.start() and match.end() <= end
                                       for start, end in (item.span() for item in CALLBACKS.finditer(line))))]
            if matches:
                print(f"{name}:{number}: {line.strip()}")
                hits += 1
    print(f"Public tree: {hits} matching lines")
    return int(hits > 0)


if __name__ == "__main__":
    sys.exit(main())
