# Live presentation runbook (EN speech → SK/CS speech) — 2026-11-04

Status: draft from measurements in `implementation-2026-11-04/streaming-findings.md`. Voice cloning is out of scope for the first show (owner, 2026-11-04); generic Piper voice is the live voice. Every number below names its source; anything else is `unverified`.

## What to expect (measured, M1 CPU, Piper, EN→SK, 45 s continuous reading)

- Mean delay from the middle of your words to the listener hearing them: **2.5 s** with the shipped EN-source pack defaults (0.25 s silence cut, 3 s max segment) and Auto speed. Worst word 4.1–4.4 s. Two repeat runs of the old 6 s setting gave 3.9 s / 3.9 s mean, so the 1.4 s gain is outside the observed 0.1 s run-to-run spread (3 runs, one recording, one speaker).
- Pause after you stop: first translated sound ~0.6–0.8 s later.
- Translated speech is 0.7–0.95× as long as your speech, so there is little slack: speak with short pauses between sentences. Non-stop reading at lecture pace saturates the queue.
- Fragmenting into 3 s pieces did not lower translation similarity to unfragmented MT of the same words (chrF++ 68 either way, one passage). Meaning was not graded by a human.
- On a Tesla T4 (GPU recognition, generic Piper, 90 s, Kaggle) the same default gave **2.1 s mean, 3.2 s worst**, repeat run identical within 0.1 s.
- Incremental word-by-word mode (`BP_INCREMENTAL=1`) was worse on CPU (5–6 s mean) and on the T4 (+0.8 to +1.6 s). Do not use it for the show unless the GPU rehearsal (see below) proves otherwise.

## Setup on the show machine

1. `python3.11 scripts/setup.py --dev`, then `make test` (exit 0 expected), `make demo-check`.
2. `make run` → open `https://localhost:8000` in **Chrome/Edge** (output-device routing needs `setSinkId`; Safari/Firefox fall back or stop). Accept the self-signed certificate once; the microphone needs the secure context.
3. Live page → Prepare EN→SK → sound check → Translation speed **Auto** → headphones or an isolated output → Start.
4. Keep `uptime` load under 6; close other heavy apps.

## Fallbacks (decide before going on stage)

- Delay creeps above ~5 s: raise Translation speed to Fastest, pause longer between sentences.
- Translation stops with "falling behind": press Start again (queue limit is 10 s of queued audio by design).
- Anything else: the YouTube demo from the README is the recorded fallback.

## Rehearsal gates (not yet run)

| Gate | Pass |
|---|---|
| 10-minute scripted talk on the show machine, real mic + real output | no crash, mean delay ≤ 3 s, worst ≤ 6 s from `scripts/live/paced_replay.py` on the same recording, Stop works at any moment |
| Native-Slovak listener grades 20 sentences | ≥ 80% understood without the slide |
| Stop / Start 20× including mid-sentence | no late audio after Stop |
| Projector room, real speakers | no feedback loop (headphones or isolated output) |

## Deployment checklist (beside LHC)

Unknown until the site visit: room input/return channels, OS, install rights, who signs off. Ask for: GPU model and VRAM, OS and admin rights, how the room audio reaches a PC, whether a virtual device (BlackHole on macOS, VB-Cable on Windows) may be installed, echo isolation of the return path. `guide.md` has the virtual-device notes; README does not yet.
