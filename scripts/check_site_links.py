#!/usr/bin/env python3
"""Check local product-page links and anchors without a server or network."""
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / 'documentation/site/index.html'


class Links(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.links, self.ids = [], set()
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.add(attrs['id'])
        for key in ('href', 'src'):
            if key in attrs:
                self.links.append(attrs[key])


def main():
    parsed = Links(PAGE.read_text(encoding='utf-8'))
    errors, checked = [], 0
    for href in parsed.links:
        url = urlsplit(href)
        if url.scheme or url.netloc:
            continue
        checked += 1
        path = (PAGE.parent / unquote(url.path)).resolve() if url.path else PAGE
        if not path.exists():
            errors.append(href)
        elif url.fragment and path.is_file():
            text = path.read_text(encoding='utf-8')
            if path.suffix == '.html':
                ids = Links(text).ids
            elif path.suffix == '.md':
                ids = {re.sub(r'[^\w\- ]', '', h.lower()).replace(' ', '-')
                       for h in re.findall(r'^#+\s+(.+)$', text, re.M)}
            else:
                ids = set()
            if unquote(url.fragment) not in ids:
                errors.append(href)
    for href in errors:
        print(f'BROKEN {href}')
    print(f'Site links: {checked} local links checked, {len(errors)} broken')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
