#!/usr/bin/env python3
"""Generate the two static manifests the UI falls back to when the server offers no capability endpoint.

    python3 scripts/ui_manifest.py          # writes ui/i18n/index.json and ui/shared/manifest.json
    python3 scripts/ui_manifest.py --check  # exit 1 if either file is stale (use in CI / make test)

Nothing here lists a language by hand: interface locales come from the files in ui/i18n/, speech routes
from languages/packs/*.json. Drop a file, rerun, the pickers show it. Names are rendered in the browser
with Intl.DisplayNames, so the manifests hold codes only.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import i18n_tool  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RTL = {"ar", "he", "fa", "ur", "ps", "sd", "yi"}


def locales():
    """Coverage is measured against the strings the pages actually contain (scripts/i18n_tool.py extract),
    not against the biggest catalog. A catalog flagged machine_draft is a preview: the picker hides it unless asked."""
    wanted = set(i18n_tool.extract())
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "ui", "i18n", "*.json"))):
        code = os.path.basename(path)[:-5]
        if code == "index":
            continue
        data = json.load(open(path, encoding="utf-8"))
        meta = data.get("_meta", {}) if isinstance(data.get("_meta"), dict) else {}
        raw, llm = set(meta.get("raw_keys", [])), set(meta.get("draft_keys", []))
        shown = sum(1 for k in data if k in wanted and k not in raw)               # what a visitor sees
        reviewed = sum(1 for k in data if k in wanted and k not in raw and k not in llm)
        n = max(len(wanted), 1)
        out.append({"code": code, "dir": "rtl" if code.split("-")[0] in RTL else "ltr", "strings": shown,
                    "coverage": round(shown / n, 2), "reviewed": round(reviewed / n, 2),
                    "machine_draft": reviewed / n < 0.5, "preview": shown / n < 0.75,
                    "drafts": len(llm & wanted), "raw": len(raw & wanted)})
    return [{"code": "en", "dir": "ltr", "strings": len(wanted), "coverage": 1.0, "reviewed": 1.0, "machine_draft": False, "preview": False, "drafts": 0, "raw": 0}] + out


def routes():
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "languages", "packs", "*.json"))):
        pack = json.load(open(path, encoding="utf-8"))
        out.append({"pair": pack["pair"], "source": pack["source"], "target": pack["target"],
                    "tts": (pack.get("tts") or {}).get("voice"), "license": pack.get("license", "unknown")})
    return out


def build():
    return {
        os.path.join("ui", "i18n", "index.json"): {"locales": locales()},
        os.path.join("ui", "shared", "manifest.json"): {"routes": routes()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    stale = False
    for rel, data in build().items():
        path = os.path.join(ROOT, rel)
        text = json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + "\n"
        old = open(path, encoding="utf-8").read() if os.path.exists(path) else None
        if args.check:
            stale |= old != text
            print(("STALE " if old != text else "ok    ") + rel)
        else:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "w", encoding="utf-8").write(text)
            print("wrote", rel)
    sys.exit(1 if stale else 0)


if __name__ == "__main__":
    main()
