# BP cross-project matrix — 2026-11-04

Status: plan, owner-approved order (chat 2026-11-04). Nothing here is a measured result unless it names a file. Deadline: thesis presentation within the next year (owner), no earlier date given.

## Tracks and gates

| Track | Goal | Gate (pass = all) | State |
|---|---|---|---|
| A. Proof of route | EN speech → SK/CS speech in the owner's cloned voice, streaming, on a T4 | Full STT→MT→cloned-TTS→playback route run by a paced client at 1x real time against the real WS server on the T4; p50/p95 speech-end→first-audio and 5-min queue growth recorded; bilingual human grade on meaning, pronunciation, identity. Proposed bound: p95 ≤ 2.5 s consecutive mode, zero queue growth over 5 min (owner to confirm). | TTS stage only measured (`kaggle-sanitized-receipt.json`) |
| A2. Reverse routes | SK→EN, CS→EN | Per route: STT WER < 0.2 on a reviewed set + meaning grade; failing routes show `experimental` in UI. Candidate: Canary ASR (SK WER 0.0477, AGENTS.md 2026-11-03) + existing MT; unverified end to end. `whisper-small-sk` is known bad (WER 0.27). CS→EN: no evidence found. | open |
| B. LHC integration | Feed translated audio through a virtual output beside LHC | Handler answers (GPU model/VRAM, OS, sudo, room input + return channels, echo isolation, acceptance owner); real-room rehearsal. | blocked on handler |
| C. Open-source product | Anyone can install, understand and run it | Clean-clone setup timed on macOS, Linux, Windows AMD; README status table per route; licence inventory (Piper GPL-3.0-or-later, OmniVoice, SeamlessM4T non-commercial); setup-doctor page; UI guidance. | docs honest, product not built |

## What the Kaggle test can prove (owner point, accepted)

Run the real FastAPI server and a paced client (recorded continuous English at 1x) in the same notebook over loopback. That exercises streaming, real STT/MT/TTS and timestamps per chunk. It does not prove a physical mic, speaker, room echo or a remote tunnel. Label the result "T4 paced replay of the full route", not "live".

## Order

1. P0 (done locally, extended): see `implementation-2026-11-04/streaming-findings.md` — barge-in, clause-sized TTS messages, ordered TTS, VAD-config bug fixed; `scripts/live/paced_replay.py` is the 1x paced harness for Mac and Kaggle.
   Original P0: event-driven playback + `playbackRate`; honest queue model in `scripts/gpu_bench/kaggle_session/job.py` with rate and drop-late levers; table in `implementation-2026-11-04/queue-model-rate-table.txt`.
2. P1: Kaggle paced full-route job (needs owner yes per submission).
3. P2: reverse routes on the T4.
4. P3 small now / large after P1: honest README + licence inventory; commit split of the dirty tree (owner approval).
5. P4: handler questions.

## Piper retrain

Deferred. Own-voice quality comes from the OmniVoice clone first; a Piper fine-tune on Kaggle is the CPU fallback and costs hours plus the 75-sentence set.
