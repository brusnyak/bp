"""The UI knows its languages from files, not from markup: generated manifests stay fresh and the live page lists none by hand."""
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_manifests_are_fresh():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "ui_manifest.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, "run `python3 scripts/ui_manifest.py`\n" + r.stdout


def test_every_dictionary_is_in_the_locale_index():
    index = {l["code"] for l in json.loads((ROOT / "ui/i18n/index.json").read_text(encoding="utf-8"))["locales"]}
    files = {p.stem for p in (ROOT / "ui/i18n").glob("*.json") if p.stem != "index"}
    assert files | {"en"} == index


def test_every_pack_is_in_the_route_manifest():
    routes = {r["pair"] for r in json.loads((ROOT / "ui/shared/manifest.json").read_text(encoding="utf-8"))["routes"]}
    packs = {p.stem for p in (ROOT / "languages/packs").glob("*.json")}
    assert packs == routes


def test_live_page_lists_no_languages_by_hand():
    html = (ROOT / "ui/live-speech/live.html").read_text(encoding="utf-8")
    for sid in ("inputLanguageSelect", "outputLanguageSelect"):
        block = re.search(r'<select\s+id="%s".*?</select>' % sid, html, re.S).group(0)
        assert block.count("<option") <= 2, f"{sid} has a hand-written language list; live-langs.js fills it from the server"
