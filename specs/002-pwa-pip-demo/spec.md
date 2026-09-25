# Feature Specification: PWA Control-Room + PiP Subtitles + Monday Demo (v3)

**Feature Branch**: `002-pwa-pip-demo`
**Created**: 2026-09-25
**Status**: Draft
**Input**: Owner direction 2026-09-25 — the "app" is a PWA control-room + floating PiP
subtitle window + virtual-mic routing, not a native shell. Demo target: Monday 2026-09-28,
owner's Mac + AMD Windows laptop, live with handler if required, both EN→SK and SK→EN.

## Vehicle decision (settled)

Native shells (Tauri/Electron/Swift) are OUT for the demo horizon. The PWA covers the
app requirements: control-room UI (existing live page), PiP floating captions across
tabs/windows/apps on the demo machine, virtual-mic device select for Zoom/Meet/Teams
audio injection. Track B native shell stays parked post-defense. Constitution I
(cross-platform, no vendor default) and III (Slovak non-negotiable) still govern.

## User Scenarios & Testing

### User Story 1 — Floating subtitles across windows (Priority: P1) 🎯 demo
Speaker runs the live page, pops subtitles into a PiP window, shares screen or sits the
PiP over Zoom/Meet. Captions follow the speech in real time on any visible window.

**Independent Test**: start session, click "pop out subtitles", switch to another app;
PiP window stays visible and updates on `caption_partial` + final transcript.
**Acceptance**: PiP launches on user gesture (browser rule); updates ≤1s after the
in-page strip; closes/reopens without breaking the WS session; unsupported browser shows
a clear message, not silence.

### User Story 2 — Monday demo runs both directions (Priority: P1) 🎯 demo
EN→SK live in owner's Hybrid-cloned voice with projected subtitles; SK→EN path proven
as far as hardware allows without the SK recording (generic Piper SK base + Hybrid
fallback), fully live once the quiet-room SK session lands.

**Independent Test**: EN script → SK audio + SK subtitles on projector; SK input →
EN output through the same chain.
**Acceptance**: offline-capable (no venue wifi dependency); pre-warmed models; backup
take on hotkey through the identical pipeline; VAD re-tuned at soundcheck.

### User Story 3 — Lab tracks the build (Priority: P2)
Voice Lab gains a plan toggle rendering `PLAN.md` live (zero drift) so owner + handler
see done/now/next without asking.

### Edge Cases
- Browser without PiP support → inline strip remains, explicit notice.
- Venue without internet → all models + voices pre-downloaded; demo runs offline.
- Mic failure on stage → backup take through the same pipeline (indistinguishable chain).
- SK recording not done by Monday → SK→EN shown via generic+Hybrid path, labelled as such.

## Requirements
- **FR-001**: PiP window MUST render live captions (partial + final) and follow across windows.
- **FR-002**: In-page subtitle strip stays (context above transcription/translation boxes).
- **FR-003**: Demo MUST run offline (pre-warmed, pre-downloaded).
- **FR-004**: Lab plan panel MUST render the repo `PLAN.md`, not a copy.
- **FR-005**: No new TTS/STT/MT engine in this feature — packaging + UI only (YAGNI).

## Success Criteria
- **SC-001**: PiP opens, updates, survives app-switching on the demo Mac AND the AMD Windows laptop (Chrome/Edge).
- **SC-002**: Monday script (5–6 sentences EN→SK) runs end-to-end live with subtitles projected.
- **SC-003**: Handler sees continued work: Lab page + plan + voices + PiP in one sitting.

## Assumptions
- Owner's voice recordings gate only SK→EN liveness and final voice QC — not PiP, plan panel, DESIGN.md, or EN→SK demo.
- Working name for the product: **LIVO** (proposed 2026-09-25; owner confirms). Full: live voice translation in your own voice, near real time.
