# Assessment B — design evidence, 2026-11-04

Method: isolated Codex reviewer `/root/design_evidence`; deterministic detector plus source inspection. Assessment A was not read. Scope: home, authentication, enrollment, live speech, Voice Lab and directly relevant serving paths. No implementation changes.

## Verdict and limits

The source has useful accessibility foundations, but does not establish a safe, predictable university meeting journey. The highest-value work is reliable routing, private voice handling and recovery, ahead of visual styling.

`curl -k -I --max-time 3 https://localhost:8000` returned exit 7, connection refused. Per packet, no backend/model was started and no alternative static server was created. Browser screenshots, computed contrast, responsive layouts, keyboard completion, microphone permissions, login, transport and playback were **not tested**. No browser tab or server was opened. No live overlay was injected; this is a source-only assessment, not a rendered visual acceptance.

## Deterministic detector receipt

Command: `/Users/yegor/.codex/skills/impeccable/scripts/impeccable detect --json ui`

Exit 2; **24 warnings**, not 24 verified defects. Counts: 12 `overused-font`, 5 `low-contrast`, and one each of `flat-type-hierarchy`, `skipped-heading`, `marketing-buzzword`, `nested-cards`, `dark-glow`, `ai-color-palette`, `border-accent-on-rounded`.

| Rule | Locations reported | Interpretation |
|---|---|---|
| overused-font (12) | `ui/auth/auth.css:45`; `ui/auth/auth.html`; `ui/enroll/enroll.html`; `ui/extra/{about,cookies,faq,feedback,privacy,terms}.html`; `ui/home/home.html`; `ui/live-speech/live.html`; `ui/voice-lab/lab.html` | Montserrat is explicitly required by current `DESIGN.md`; suppress as non-actionable aesthetic preference unless owner changes design authority. Detector description names other fonts, while snippets flag Montserrat. |
| low-contrast (3) | `ui/home/home.html` | Source estimates 1.2:1 `#333333/#19252c`; 2.7:1 `#c41e3a/#19252c`; 2.7:1 `#666666/#19252c`. Confirm actual element/background/theme in browser before accepting. |
| low-contrast (2) | `ui/test-stt/test-stt.html` | Source estimates 3.7:1 `#ffffff/#667eea`; 2.2:1 `#a0aec0/#f7fafc`. Developer test surface; prioritize below main journey. |
| flat-type-hierarchy | `ui/enroll/enroll.html` | Detector estimates body/h1/h2 all 16px; rendered CSS inheritance must be checked. |
| skipped-heading | `ui/extra/faq.html` | h1→h3 reported; source outline correction is inexpensive. |
| marketing-buzzword | `ui/home/home.html` | “optimizing cutting-edge open-source”; replace with concrete capability and evidence. |
| nested-cards | `ui/live-speech/live.html` | Structural styling heuristic; not evidence of task failure. |
| dark-glow | `ui/live-speech/live.html` | Zero-offset glow `#2e7d32`; styling preference, lower priority than usability. |
| ai-color-palette | `ui/test-stt/test-stt.html` | Purple accents on developer surface; low priority. |
| border-accent-on-rounded | `ui/voice-lab/lab-style.css:97` | `border-top: 3px solid`; styling preference, lower priority. |

For detector HTML findings, emitted line `0` means no usable source position. Do not cite these as line 1 or as browser-computed WCAG results.

## Source-confirmed issues and acceptance checks

1. **P1 — Voice evidence is publicly addressable on any reachable deployment.** `app.py:201` mounts all `/ui`, including `ui/voice-lab/library.json`; `app.py:203` mounts speaker clips, `app.py:206` unrestricted QC static files, and `app.py:210–214` selected processed audio. `_AudioOnlyStatic` (`app.py:48–54`) restricts extensions but has no identity/owner check. `lab.js:902` fetches the library without auth and `lab.js:234,247` renders transcripts. A collapsed “Show recordings” panel (`lab.html:63–72`) is presentation control, not access control. Existing loopback deployment limits reachability; this is not proof of an internet exposure. **Fix:** separate approved public demo assets from private user recordings and authorize private media/metadata per user. **Accept:** unauthenticated and second-user requests cannot fetch private clip URLs or metadata; approved demo still plays.
2. **P1 — Routing is started asynchronously while playback proceeds immediately.** `live-script.js:411–424` starts `setSinkId()` promise; `live-script.js:436` calls `audio.play()` without awaiting route completion. A default-speaker burst is a plausible race, not reproduced. On unsupported `setSinkId`, a selected device falls through to default playback (`425–426`), logged only to console. **Fix:** resolve device routing before playback; when meeting output is selected, give a visible choice before fallback to speakers. **Accept:** record output channels with delayed/failed/unsupported routing and verify no unintended default playback.
3. **P1 — Token disclosure in developer console.** `ui/auth/auth.js:48` logs the returned bearer token plus identity after password login. **Fix:** remove token logging. **Accept:** login with a synthetic token and assert console contains no token. This is source-confirmed behavior, not an exploited account.
4. **P1 — Enrollment recovery loses its retry handler on network exceptions.** `enroll.js:73` binds submit with `{once:true}`; `108–140` handles HTTP failure but `fetch` network rejection has no enclosing catch. The explicit retry listener is restored only for non-OK upload responses (`118–122`). **Fix:** make submission reentrant with disabled/loading state, catch transport failures and keep a resumable list; avoid restarting all accepted uploads. **Accept:** interrupt upload and processing separately; retry completes from retained takes without duplicate clips. Refresh currently discards in-memory takes (`40–43`); warn before leaving or persist recoverable progress.
5. **P2 — Privacy wording contradicts available upload action.** Voice Lab says “nothing uploads anywhere” (`lab.html:108`), while `lab.js:759–769` posts files to the backend through an explicit upload button (`lab.js:810+`). Upload is user-triggered, not automatic; label the storage destination and consent before the button. **Accept:** staging remains local; explicit upload clearly states server storage and shows completion/recovery.
6. **P2 — Keyboard and screen-reader gaps in entry flow.** Home hamburger is a `<div>` (`home.html:26`) with click handler only (`home.js:36–40`); section navigation uses clickable divs (`home.html:125+`). Authentication feedback container lacks status/live semantics (`auth.html:79`); enrollment session name lacks a label (`enroll.html:40`). Naming modal close is a span (`live.html:470`). **Fix:** native buttons/links and labelled/status semantics, then verify focus order, visible focus, dialog trapping/escape and announcements in browser. **Accept:** keyboard-only home→login→session→stop and a screen-reader recovery pass.
7. **P2 — Research engines appear in default operation controls.** Live engine list (`live.html:149–155`) exposes OmniVoice “untested live” and XTTS beside ordinary options. Labels are more honest than hiding uncertainty, but availability/license/route readiness should be resolved before selection. **Accept:** ordinary user sees a supported preset; experimental engines require an explicit advanced opt-in with the actual block reason.
8. **P2 — Help cannot resolve a privacy request through its own link.** `ui/extra/privacy.html:64,69` links `/ui/feedback.html`; the file inventory contains `ui/extra/feedback.html`. **Fix:** correct route as part of help-link cleanup. **Accept:** all primary help/recovery links resolve in the deployed mount.

## Existing strengths

- Live interface has labelled output/language controls, mic meter semantics and polite state/subtitle regions (`live.html:95–141,158–180,218+`).
- Output device enumeration preserves default speakers rather than silently moving audio to a detected virtual cable (`live-script.js:93+`); visible playback-error notifications exist (`416–442`).
- Enrollment stops mic tracks after each take (`enroll.js:87–88`), allows playback/re-record, and states training requires approval (`138–139`).
- Voice Lab distinguishes saved evidence from optional live mic proof (`lab.html:60`) and labels staging state (`lab.js:800`).

## Suggested execution order

1. Protect private artifacts and remove token logging before LAN/university exposure.
2. Establish one supported route preset and deterministic output routing, with a pre-meeting sound check and recovery.
3. Fix enrollment upload recovery and truthful storage copy; keep voice cloning optional to first translation.
4. Run rendered desktop/projector/mobile, keyboard and screen-reader verification; only then accept contrast findings and prioritize visual refinement.

Weakest point: a presentable evidence room can be mistaken for an authorized product boundary. Cheapest failure: a known recording URL opens without login on a reachable deployment. Smallest fix: approved public demo manifest plus protected private assets, then a two-user access test.

Questions skipped: parent orchestrator owns user clarification and final synthesis. Token usage: tool surface did not expose per-agent accounting.
