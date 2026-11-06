#!/usr/bin/env python3
"""Interface translation workflow. English text is the key (gettext style); catalogs live in ui/i18n/<code>.json.

    python3 scripts/i18n_tool.py extract            # every UI string found in ui/**/*.html and ui/**/*.js (t(), T(), N_())
    python3 scripts/i18n_tool.py missing sk         # strings the sk catalog does not have yet
    python3 scripts/i18n_tool.py coverage           # per-locale coverage against the extracted set
    .venv/bin/python scripts/i18n_tool.py draft cs  # fill the gaps with the project's own Opus-MT model (en -> cs): rough, hidden by default
    ~/.claude/scripts/with-env NIM_3 -- python3 scripts/i18n_tool.py translate cs   # LLM pass (NVIDIA NIM, free tier): shown, labelled "Machine draft"

Catalog `_meta` tracks provenance per string: `raw_keys` = raw Opus-MT output, hidden from visitors unless ?drafts=1;
`draft_keys` = LLM translations, shown and labelled "Machine draft" in the picker. A reviewer fixes a string and removes
it from either list. Only public interface text is ever sent to the LLM (legal pages are excluded).
Strings with placeholders or markup are never machine-drafted (they stay English until a person translates them).
"""
import argparse
import datetime
import glob
import html.parser
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
I18N = os.path.join(ROOT, "ui", "i18n")
SKIP_TAGS = {"script", "style", "code", "pre", "noscript"}
# Legal pages stay in one reviewed language until a lawyer checks a translation (DESIGN.md, change protocol 6).
LEGAL = ("/extra/privacy.html", "/extra/terms.html", "/extra/cookies.html")
ATTRS = {"aria-label", "title", "alt", "placeholder"}
ICON = re.compile(r"^[a-z_]+$")
CALL = re.compile(r"""\b(?:t|T|N_|i18n\.t)\(\s*(['"])((?:\\.|(?!\1).)*)\1""")


class _Text(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.out = [], []

    def handle_starttag(self, tag, attrs):
        self.stack.append(tag)
        d = dict(attrs)
        if "translate" in d and d["translate"] == "no":
            self.stack.append("__no__")
        for k, v in attrs:
            if k in ATTRS and v:
                self.out.append(v)

    def handle_endtag(self, tag):
        while self.stack:
            if self.stack.pop() == tag:
                break

    def handle_data(self, data):
        if any(t in SKIP_TAGS or t == "__no__" for t in self.stack):
            return
        n = " ".join(data.split())
        if n and re.search(r"[A-Za-z]{2,}", n) and not ICON.match(n) and n[0].isalnum() and not re.fullmatch(r"[A-Z]{2}|[\d.]+\s?s", n):
            self.out.append(n)


def extract():
    found = {}
    for path in glob.glob(os.path.join(ROOT, "ui", "**", "*.html"), recursive=True):
        if "/vendor/" in path or "/voice-lab/" in path or "/test-stt/" in path or path.endswith(LEGAL):
            continue
        p = _Text()
        p.feed(open(path, encoding="utf-8").read())
        for s in p.out:
            found.setdefault(s, set()).add(os.path.relpath(path, ROOT))
    for path in glob.glob(os.path.join(ROOT, "ui", "**", "*.js"), recursive=True):
        if "/vendor/" in path or "/voice-lab/" in path or "/test-stt/" in path:
            continue
        for m in CALL.finditer(open(path, encoding="utf-8").read()):
            s = " ".join(bytes(m.group(2), "utf-8").decode("unicode_escape", errors="ignore").split()) if "\\" in m.group(2) else " ".join(m.group(2).split())
            if s and re.search(r"[A-Za-z]", s):
                found.setdefault(s, set()).add(os.path.relpath(path, ROOT))
    return found


def load(code):
    path = os.path.join(I18N, code + ".json")
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}


def missing(code):
    cat = load(code)
    return sorted(s for s in extract() if s not in cat)


def codes():
    return sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(I18N, "*.json")) if not p.endswith("index.json"))


def draftable(s):
    return bool(re.search(r"[A-Za-z]{2,}", s)) and not re.search(r"[{}<>@]|https?://|\.(py|md|json|html)\b", s) and len(s) < 400


def draft(code):
    import ctranslate2  # project venv
    from transformers import AutoTokenizer
    model = os.path.join(ROOT, "ct2_models", f"Helsinki-NLP--opus-mt-en-{code}")
    if not os.path.exists(os.path.join(model, "model.bin")):
        sys.exit(f"No local Opus-MT model for en -> {code} ({model}). Convert one with scripts/convert_models.py first.")
    cat = load(code)
    gaps = missing(code)
    todo = [s for s in gaps if draftable(s)]
    print(f"{code}: {len(todo)} to draft, {len(gaps) - len(todo)} left in English (placeholders/markup/non-prose)")
    tok = AutoTokenizer.from_pretrained(model)
    tr = ctranslate2.Translator(model, device="cpu", intra_threads=int(os.environ.get("HEAVY_THREADS", "3")))
    src = [tok.convert_ids_to_tokens(tok.encode(s)) for s in todo]
    results = tr.translate_batch(src, beam_size=4, max_batch_size=24)   # one batched call: far faster than string by string
    meta = cat.get("_meta", {}) if isinstance(cat.get("_meta"), dict) else {}
    keys = set(meta.get("raw_keys", []))
    for s, r in zip(todo, results):
        out = tok.decode(tok.convert_tokens_to_ids(r.hypotheses[0]), skip_special_tokens=True).strip()
        if out and out.lower() != s.lower():
            cat[s] = out
            keys.add(s)
    meta.update({"raw_keys": sorted(keys), "generated": datetime.date.today().isoformat()})
    cat["_meta"] = meta
    open(os.path.join(I18N, code + ".json"), "w", encoding="utf-8").write(json.dumps(cat, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote ui/i18n/{code}.json with {len(keys)} raw machine strings; run scripts/ui_manifest.py")


PRODUCT_TERMS = "Piper, Opus-MT, CTranslate2, faster-whisper, WebSocket, BlackHole, VB-Cable, GPU, CPU, Mac M1, Tesla T4, OmniVoice, STU FEI, Chrome, Edge, ffmpeg, Python, GitHub"
NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"


def _nim(model, system, user, key, timeout=180):
    import urllib.request
    body = {"model": model, "temperature": 0.2, "max_tokens": 6000,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if "nemotron" in model:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(NIM_URL, data=json.dumps(body).encode(), headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.loads(f.read())["choices"][0]["message"].get("content") or ""


def _json_object(text):
    a, b = text.find("{"), text.rfind("}")
    return json.loads(text[a:b + 1]) if a != -1 and b > a else {}


def translate(code, model, key_env, only, batch=30):
    import time
    key = os.environ.get(key_env, "")
    if not key:
        sys.exit(f"Set {key_env}: run through ~/.claude/scripts/with-env {key_env} -- python3 scripts/i18n_tool.py translate {code}")
    try:
        lang = __import__("locale")  # noqa: F841
        from babel import Locale  # optional
        name = Locale.parse(code).get_display_name("en")
    except Exception:
        name = {"cs": "Czech", "pl": "Polish", "uk": "Ukrainian", "sk": "Slovak", "de": "German", "hu": "Hungarian", "es": "Spanish", "fr": "French"}.get(code, code)
    cat = load(code)
    meta = cat.get("_meta", {}) if isinstance(cat.get("_meta"), dict) else {}
    raw, drafts = set(meta.get("raw_keys", [])), set(meta.get("draft_keys", []))
    wanted = [s for s in extract() if re.search(r"[A-Za-z]{2,}", s) and len(s) < 400 and not re.search(r"[<>@]|https?://|\.(py|md|json|html)\b", s)]   # an LLM can keep {placeholders}; Opus-MT cannot
    todo = [s for s in wanted if s not in cat or (only == "raw" and s in raw)]
    print(f"{code} ({name}): {len(todo)} strings to translate with {model}")
    system = (f"You localize the interface of a live speech-translation web app into {name} ({code}). "
              "Translate each English UI string. Natural, concise, neutral-formal register (use the polite form of 'you' where the language has one). "
              f"Keep these terms and product names unchanged: {PRODUCT_TERMS}. Keep numbers, units, trailing punctuation and capitalisation style. "
              "'Start'/'Stop' are button verbs. A 'route' is a language-pair translation direction. "
              "Some strings are fragments of a sentence that continues after a link: translate the fragment as a fragment. Keep {placeholders} exactly. "
              "A string ending in ' [few]' or ' [many]' is a plural-category variant: use the grammatical plural form of that category in the target language and drop the marker. "
              "Reply with ONE JSON object and nothing else: keys are the exact English strings, values the translations.")
    done = 0
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        got = {}
        for attempt in range(2):
            need = [c for c in chunk if c not in got]
            if not need:
                break
            try:
                reply = _nim(model, system, json.dumps(need, ensure_ascii=False), key)
                for k, v in _json_object(reply).items():
                    if k in need and isinstance(v, str) and v.strip() and sorted(re.findall(r"\{\w+\}", k)) == sorted(re.findall(r"\{\w+\}", v)):
                        got[k] = v.strip()
            except Exception as e:  # network, 429, bad JSON: retry the missing ones once
                print("  retry:", type(e).__name__, str(e)[:80])
            time.sleep(2.6)   # stay under the shared 24 requests/minute NIM limit
        for k, v in got.items():
            cat[k] = v; drafts.add(k); raw.discard(k)
        done += len(got)
        print(f"  {min(i + batch, len(todo))}/{len(todo)} ({len(got)} ok in this batch)")
    meta.update({"raw_keys": sorted(raw), "draft_keys": sorted(drafts), "engine_llm": model, "generated": datetime.date.today().isoformat()})
    cat["_meta"] = meta
    open(os.path.join(I18N, code + ".json"), "w", encoding="utf-8").write(json.dumps(cat, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote ui/i18n/{code}.json: {done} LLM strings ({len(todo) - done} still missing); run scripts/ui_manifest.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["extract", "missing", "coverage", "draft", "translate"])
    ap.add_argument("code", nargs="?")
    ap.add_argument("--model", default="nvidia/nemotron-3-ultra-550b-a55b")
    ap.add_argument("--key-env", default="NIM_3")
    ap.add_argument("--only", choices=["missing", "raw"], default="missing", help="raw = also redo strings that only have raw Opus-MT output")
    a = ap.parse_args()
    if a.cmd == "extract":
        for s, where in sorted(extract().items()):
            print(f"{s}\t{','.join(sorted(where))}")
    elif a.cmd == "missing":
        for s in missing(a.code):
            print(s)
    elif a.cmd == "coverage":
        total = len(extract())
        for c in codes():
            cat = load(c)
            have = sum(1 for s in extract() if s in cat)
            m = cat.get("_meta") or {}
            print(f"{c}: {have}/{total} ({have * 100 // total}%)  llm drafts: {len(m.get('draft_keys', []))}  raw machine: {len(m.get('raw_keys', []))}")
    elif a.cmd == "draft":
        sys.path.insert(0, ROOT)
        draft(a.code)
    elif a.cmd == "translate":
        translate(a.code, a.model, a.key_env, a.only)


if __name__ == "__main__":
    main()
