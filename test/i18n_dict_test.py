"""Interface catalogs: valid JSON, no stray whitespace (a trailing space once made the page loop forever),
placeholders preserved, machine drafts declared, shipped locales cover most of the page."""
import json
import re
from pathlib import Path

I18N = Path(__file__).resolve().parent.parent / "ui" / "i18n"
CODES = sorted(p.stem for p in I18N.glob("*.json") if p.stem != "index")


def _load(code):
    return json.loads((I18N / f"{code}.json").read_text(encoding="utf-8"))


def test_catalogs_are_clean():
    assert CODES, "no catalogs found"
    for code in CODES:
        for en, tr in _load(code).items():
            if en == "_meta":
                continue
            assert en and tr, (code, en)
            assert tr == tr.strip(), f"{code}: translation of {en!r} has leading/trailing whitespace"
            assert sorted(re.findall(r"\{\w+\}", en)) == sorted(re.findall(r"\{\w+\}", tr)), (code, en)


def test_declared_machine_keys_exist():
    for code in CODES:
        d = _load(code)
        for listed in ("raw_keys", "draft_keys"):
            for key in d.get("_meta", {}).get(listed, []):
                assert key in d, f"{code}: {listed} names {key!r} which has no translation"


def test_shipped_locales_cover_the_pages():
    index = {l["code"]: l for l in json.loads((I18N / "index.json").read_text(encoding="utf-8"))["locales"]}
    for code, loc in index.items():
        if not loc["preview"]:
            assert loc["coverage"] >= 0.75, f"{code} is offered in the picker but covers {loc['coverage']:.0%} of the page text"
