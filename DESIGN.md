# DESIGN.md — LIVO visual language (house style)

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

## Page contracts

- **Home** (`ui/home/`): marketing + thesis-mode sections, fixed nav.
- **Live** (`ui/live-speech/`): control room. Subtitle strip above the columns (dark band,
  large type, projector-readable) + optional PiP pop-out mirroring it.
- **Voice Lab** (`ui/voice-lab/`): static, backend-free. Reads `library.json` only.
  Regenerate data with `python3 scripts/update_voice_lab_library.py`, never hand-edit output.

## Product name

Shortlist (owner decides): **Hlas** (SK "voice" — recommended: short, own, handler-friendly) ·
LIVO · VoxBridge. Manifest + icons currently ship Hlas. Use the chosen name in titles,
demo scripts, and handler-facing material. (Legacy "Lingonberry" footer stays until renamed.)
