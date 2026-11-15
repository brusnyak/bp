#!/usr/bin/env python3
"""Design-system gate for the Live and Models pages (stdlib only).

Fails (exit 1) when
  1. page CSS uses a literal color, spacing/type/radius length, duration or z-index instead of a var(--bp-*) token;
  2. a token pair in ui/tokens.css misses WCAG contrast in light or dark (text 4.5, UI/large 3.0);
  3. a page that loads page CSS does not load ui/tokens.css.
Run: python3 scripts/lint_tokens.py   (also: make check-ui)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
UI = ROOT / "ui"
SCOPED_CSS = ["app.css", "live-speech/live-style.css", "models/models.css"]
SCOPED_HTML = {"live-speech/live.html": "live-speech/live-style.css", "models/models.html": "models/models.css"}
# (foreground token, background token, minimum ratio)
PAIRS = [
    ("text", "bg", 4.5), ("text", "surface", 4.5), ("muted", "bg", 4.5), ("muted", "surface", 4.5),
    ("accent", "bg", 4.5), ("accent", "surface", 4.5), ("on-accent", "accent", 4.5),
    ("on-danger", "danger", 4.5), ("success", "surface", 4.5), ("warning", "surface", 4.5),
    ("danger", "surface", 4.5), ("control", "surface", 3.0), ("control", "bg", 3.0),
]
LEN_PROPS = r"(font-size|gap|row-gap|column-gap|padding(?:-[a-z]+)?|margin(?:-[a-z]+)?|border-radius)"
RULES = [
    (re.compile(r"#[0-9a-fA-F]{3,8}\b"), "hex color"),
    (re.compile(r"\brgba?\("), "rgb()/rgba() literal"),
    (re.compile(r"\bhsla?\("), "hsl() literal"),
    (re.compile(LEN_PROPS + r"\s*:[^;{}]*?(?<![\w.-])-?(?:\d*\.?\d+(?:rem|em)|(?:[3-9]|\d{2,})px)\b"), "length literal (use --bp-s*/--bp-t-*/--bp-r-*)"),
    (re.compile(r"(?:transition|animation)[a-z-]*\s*:[^;{}]*?(?<![\w.-])\d*\.?\d+m?s\b"), "duration literal (use --bp-dur*)"),
    (re.compile(r"z-index\s*:\s*-?\d+"), "z-index literal (use --bp-z-*)"),
]


def lum(h):
    h = h.lstrip("#")
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = map(f, rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ratio(a, b):
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def theme_tokens(css):
    light = dict(re.findall(r"--bp-([a-z-]+):\s*(#[0-9A-Fa-f]{6})", css.split("[data-theme")[0]))
    dark = dict(light)
    dark.update(re.findall(r"--bp-([a-z-]+):\s*(#[0-9A-Fa-f]{6})", css.split("[data-theme", 1)[1]))
    return {"light": light, "dark": dark}


def main():
    fails = []
    tokens_css = (UI / "tokens.css").read_text()
    for name, toks in theme_tokens(tokens_css).items():
        for fg, bg, need in PAIRS:
            if fg not in toks or bg not in toks:
                fails.append(f"tokens.css {name}: missing token for pair {fg}/{bg}")
                continue
            r = ratio(toks[fg], toks[bg])
            if r < need:
                fails.append(f"contrast {name}: {fg} {toks[fg]} on {bg} {toks[bg]} = {r:.2f} < {need}")
    for rel in SCOPED_CSS:
        text = re.sub(r"/\*.*?\*/", "", (UI / rel).read_text(), flags=re.S)
        for n, line in enumerate(text.splitlines(), 1):
            for rx, why in RULES:
                if rx.search(line):
                    fails.append(f"ui/{rel}: {why}: {line.strip()[:90]}")
    for html, css in SCOPED_HTML.items():
        if "/ui/tokens.css" not in (UI / html).read_text():
            fails.append(f"ui/{html}: does not load /ui/tokens.css")
    for f in fails:
        print("FAIL", f)
    print(f"lint_tokens: {len(fails)} problem(s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
