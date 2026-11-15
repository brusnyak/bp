#!/usr/bin/env python3
"""Layout gate for Live and Models: measures the rendered page, does not read CSS.
Needs the app running (BP_URL, default http://127.0.0.1:8000) and `pip install playwright` + chromium.
Fails (exit 1) on: horizontal overflow; header content edge != page content edge; controls in one row with
different height or centre line; touch targets < 40 px; theme/language controls not in the header; text clipped.
Run: python3 scripts/check_ui_layout.py   (also: make check-layout)
"""
import os
import sys
from playwright.sync_api import sync_playwright

BASE = os.environ.get("BP_URL", "http://127.0.0.1:8000")
PAGES = {"live": "/ui/live-speech/live.html", "models": "/ui/models/models.html"}
JS = """() => {
  const out = [];
  const r = (el) => el.getBoundingClientRect();
  const vis = (el) => { const b = r(el); const s = getComputedStyle(el); return b.width > 0 && b.height > 0 && s.visibility !== 'hidden' && s.display !== 'none'; };
  if (document.documentElement.scrollWidth > innerWidth + 1) out.push('overflow-x: scrollWidth ' + document.documentElement.scrollWidth + ' > ' + innerWidth);
  const head = document.querySelector('header'), main = document.querySelector('main');
  if (head && main) {
    const lead = [...head.querySelectorAll('a, button')].filter(vis)[0], h1 = main.querySelector('h1');
    if (lead && h1 && Math.abs(r(lead).left - r(h1).left) > 2) out.push('header content edge ' + Math.round(r(lead).left) + ' != page content edge ' + Math.round(r(h1).left));
    const trail = [...head.querySelectorAll('a, button')].filter(vis).pop();
    const mainRight = r(main).right - parseFloat(getComputedStyle(main).paddingRight);
    if (trail && Math.abs(r(trail).right - mainRight) > 2) out.push('header right edge ' + Math.round(r(trail).right) + ' != page content right edge ' + Math.round(mainRight));
  }
  if (head && !head.querySelector('#theme-toggle')) out.push('theme toggle is not in the header');
  if (head && !head.querySelector('[id*="lang" i], .lang-picker, [data-i18n-picker], #uiLang, button[aria-label*="language" i]')) out.push('interface language control is not in the header');
  // sibling controls in one row must share height and centre line
  document.querySelectorAll('.row, .control-row, .direction-row, .pair-selects, .mic-device, .land-filter, .room-create').forEach((row) => {
    const kids = [...row.children].filter(vis).map((k) => k.matches('button, select, input, a') ? k : (k.querySelector('select, button, input') || null)).filter((k) => k && vis(k));
    if (kids.length < 2) return;
    const rows = {};
    kids.forEach((k) => { const key = Math.round(r(k).top / 40); (rows[key] = rows[key] || []).push(k); });
    Object.values(rows).forEach((g) => {
      if (g.length < 2) return;
      const hs = g.map((k) => Math.round(r(k).height)), cs = g.map((k) => Math.round(r(k).top + r(k).height / 2));
      if (Math.max(...hs) - Math.min(...hs) > 2) out.push((row.className || row.tagName) + ': row items differ in height ' + hs.join('/'));
      const bs = g.map((k) => Math.round(r(k).bottom));
      if (Math.max(...cs) - Math.min(...cs) > 2 && Math.max(...bs) - Math.min(...bs) > 2) out.push((row.className || row.tagName) + ': row items not aligned (centres ' + cs.join('/') + ', bottoms ' + bs.join('/') + ')');
    });
  });
  document.querySelectorAll('button, a.cta, select, input:not([type=hidden]):not([type=checkbox])').forEach((el) => {
    if (!vis(el)) return;
    const b = r(el);
    if (b.height < 40) out.push('touch target < 40 px: ' + (el.id || el.className || el.tagName) + ' (' + Math.round(b.height) + ')');
  });
  document.querySelectorAll('h1,h2,h3,p,label,span,button').forEach((el) => {
    if (!vis(el) || el.closest('.visually-hidden') || el.classList.contains('visually-hidden') || getComputedStyle(el).overflow === 'visible') return;
    if (el.scrollWidth > el.clientWidth + 2 && getComputedStyle(el).textOverflow !== 'ellipsis') out.push('clipped text: ' + (el.textContent || '').trim().slice(0, 30));
  });
  return [...new Set(out)];
}"""


def main():
    problems = 0
    with sync_playwright() as p:
        b = p.chromium.launch()
        for name, path in PAGES.items():
            for w in (1440, 390):
                pg = b.new_page(viewport={"width": w, "height": 900})
                pg.goto(BASE + path, wait_until="networkidle")
                pg.wait_for_timeout(800)
                for msg in pg.evaluate(JS):
                    print(f"FAIL {name}@{w}: {msg}")
                    problems += 1
                pg.close()
        b.close()
    print(f"check_ui_layout: {problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
