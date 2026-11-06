# DESIGN.md — Real-Time Speech Translation visual language (house style)

Single source for every UI surface in this repo (home, live, voice-lab, future PWA).
Check this file before adding any page or restyling; match it first, distinctive
choices second. Built from `ui/global-styles.css` (the contract) + the
`frontend-design` skill (the taste).

## Tokens (from `ui/global-styles.css` — never hardcode hex in page CSS)

| Role | Var | Value |
|---|---|---|
| Brand | `--primary-color` | `#C41E3A` Lingonberry (light) / `#FF4081` (dark) |
| Brand deep | `--deep-berry` / `--primary-color-dark` | `#8A0F24` |
| Ink | `--text-color` | `#333333` / dark theme light |
| Muted | `--secondary-color` | `#666666` / `#A0A0A0` |
| Surface | `--surface-color` | `#FFFFFF` / dark surface |
| Line | `--border-color` | `#E0E0E0` |
| Hero band | `--section-bg-hero` | dark-blue `#0A1C4F`-family, both themes |
| Success / danger | `--log-success-color` / `--danger-color` | green / red |

## Typography

- Headings: `var(--font-heading)` (Playfair Display, serif). Body: `var(--font-body)` (Montserrat, sans-serif).
- Never ship Inter-default or gradient-purple clichés (skill rule).

## Components

- **Hero band**: solid `var(--section-bg-hero)`, white Playfair title, no gradients. (The one
  gradient ever shipped here — voice-lab 2026-09-24 — is banned retroactively.)
- **Cards**: `var(--surface-color)`, `1px var(--border-color)`, `border-radius: 1rem`,
  `box-shadow: 0 5px 15px var(--shadow-color)`, padding `1.1rem 1.35rem`.
- **Pills/chips**: `999px` radius, `0.8rem` bold text, tinted bg + colored text.
- **Theme**: every page includes `../theme-toggle.js` + a `#theme-toggle` button containing
  `.material-symbols-outlined`; honor `[data-theme="dark"]` via vars only, never a second palette.
- **Audio**: `<audio controls>` full-width inside cards.
- **A11y minimum**: `aria-live="polite"` on live text, labelled controls, keyboard-reachable buttons.

## Page contracts (updated 2026-11-05)

- **Shell** (`ui/site.css`): one top bar (brand, section links, Sign in, theme), one footer, document-page
  layout, buttons (`.cta`), reveal motion (`.rv`, hidden only when `html.js`, off under `prefers-reduced-motion`).
  Fonts load from Google Fonts (Montserrat, Playfair Display, Material Symbols); no other third-party CSS/JS.
- **Home** (`ui/home/`, rebuilt 2026-11-05): Persuade surface, six anchors: hero with the recorded caption pair (voice line),
  Hear it (three real sentences + video), How it works (tappable stages + one latency bar chart), Get started (browser /
  your machine / meeting tabs, copy buttons, agent prompt), About (works / experimental / not yet) and FAQ (two columns on
  wide screens, boxed +/-). One translucent top bar, left section rail on 1280 px+, no section tints, no burger on wide
  screens. Every number cites a file under `documentation/implementation-2026-11-04/`. Page height 4.4k px desktop, 5.4k phone (was 10.1k / 15.9k).
- **Live** (`ui/live-speech/`): Operate surface. Slim header (menu, name, language, theme), a quiet status row, languages as
  You speak / swap / You hear (picker, filled from the server, starting pair from link, last use, then `navigator.languages`),
  one guided button (Prepare route, Check sound, I'm using headphones, Start, Stop) that forwards to the real controls so every
  safety gate stays, subtitle band above the columns. Engine experiments stay under Advanced.
- **Document pages** (`ui/extra/`): privacy, terms, cookies (with contents list), about, faq, feedback. Plain
  statements only; claims must match code (cookies page is checked against `localStorage` keys).
- **Voice Lab** (`ui/voice-lab/`): development only. Served by the app only when `BP_LAB=1`; `make lab` serves it
  standalone on 127.0.0.1. Not in the service-worker cache, not linked from public pages.

## Product name

**Real-Time Speech Translation** (owner decision 2026-11-05: keep the name). The top bar shows "Speech Translation".
The manifest and Voice Lab still say "Hlas"; rename them only if the owner asks.

## Change protocol (added 2026-11-05 after a redesign the owner rejected as generic)

1. **Purpose first.** Every page states the product in one sentence taken from `PRODUCT.md`:
   live translation in the speaker's own voice, English and Slovak in both directions. A page that could
   describe any speech tool has failed.
2. **Keep-list before any edit.** List what the owner already likes on each page (left section rail, boxed
   +/- FAQ cards, footer with signature and social icons, feature cards, latency chart) and keep it unless the
   owner says remove it. Removal is a question, not a side effect of a refactor.
3. **Baseline pair.** Screenshot the current and the previous version (git tag or backup) at 1440 and 500 px
   before touching code, and compare them side by side after. No "before" means no redesign.
4. **One signature motif.** (Shipped 2026-11-05: hero line that follows the visitor's microphone locally, a small wave mark under every section title, language chips read from the server.) The voice line: a single berry waveform that runs from the speaker's English words
   to the listener's Slovak words. It appears in the hero, the section rail and the latency chart. No stock
   landing-page template (split hero, three equal cards) as the base.
5. **Claims ledger.** Every claim on a public page has a row in `documentation/claims.md`
   (claim, status proven or target, evidence file, copy to show while it is not proven). Persuasive framing is
   allowed: lead with the best true number and name the goal as a target. An unmeasured claim is never stated as fact.
6. **Languages.** Nothing lists a language by hand. Interface locales come from `ui/i18n/*.json` (`scripts/ui_manifest.py` writes `ui/i18n/index.json`); speech routes from the server (`ui/shared/caps.js`: `/api/capabilities`, else `/api/routes`, else `ui/shared/manifest.json`). One picker (`ui/shared/lang-picker.js`) for both. Names are rendered with `Intl.DisplayNames` and marked `translate="no"`. New copy goes through keys (en, sk, de) from the start. Legal pages stay in one reviewed language until a lawyer checks a translation.
7. **Owner check.** The owner views the desktop and phone screenshots of home, live and one document page
   before the change is called done.
