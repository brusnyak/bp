# Plan: PWA Control-Room + PiP Subtitles + Monday Demo (002)

**Spec**: `specs/002-pwa-pip-demo/spec.md` | **Date**: 2026-09-25
**Vehicle**: PWA (control-room page + PiP overlay + virtual-mic routing). No native shell.
**Demo**: Mon 2026-09-28, owner's Mac + AMD Windows laptop, offline-capable, both EN→SK and SK→EN.

## Approach

Packaging + UI only. Zero model changes: the STT→MT→TTS chain, Hybrid cloning, and the
non-blocking pipeline from spec 001 are taken as-is. Every item below is additive to the
existing live page or static files — no pipeline refactor, no new engine (YAGNI).

1. **PiP subtitles (US1)**: canvas-fed hidden `<video>` → `requestPictureInPicture()`,
   redrawn on `caption_partial` + final transcript/translation. User-gesture launch,
   graceful notice where unsupported. In-page strip stays as context + fallback.
2. **Monday demo (US2)**: pre-warmed models, offline bundle check, backup take on hotkey,
   soundcheck VAD procedure, 5–6 sentence EN script. SK→EN live only after the SK recording;
   until then generic+Hybrid path, explicitly labelled.
3. **Lab plan panel (US3)**: toggle fetching repo `PLAN.md` as text — zero drift by construction.

## Risks

- PiP unsupported/shaky on the AMD laptop browser → verify Chrome/Edge beforehand; strip remains.
- Venue wifi absent → pre-download pass (models, voices, checkpoints) before Monday.
- No SK recording by Monday → SK→EN demo runs degraded path; recording session is the mitigation.

## Test plan

- Static serve: lab page 200, `library.json` + `PLAN.md` fetchable, wavs stream.
- Live boot: `/api/voice-lab/status`, lab, live pages 200; `hardware_test.py` 7/7.
- Manual: PiP opens/updates/survives app-switch on Mac + Windows; strip + PiP agree.
