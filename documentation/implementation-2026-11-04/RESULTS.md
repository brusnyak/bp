# BP repair results — 2026-11-04

## Verdict

Reviewed configuration, authentication, voice-selection, lifecycle and evidence defects were repaired. The full cloned-speech meeting product is **not commercially ready** and LHC integration remains an audio-adapter assumption. This is a local uncommitted repair batch, not a deployment or accepted university integration.

## Implemented

- WS authentication precedes session/model allocation; explicit configuration prepares models. HTTP preparation is authenticated and scoped. Missing assets stay local-only with actionable setup errors. Optional partial-caption resources require explicit local provisioning.
- Target-language Piper selection and CTranslate2 device selection corrected. Czech→English uses an English voice. Shared SDK configurations distinguish voice/beam settings.
- Stop cancels session work and queued playback, blocks stale output/resumed capture, and keeps capture active during translated speech under explicit headphone/isolated-routing confirmation. Native worker inference cannot be force-killed; its stale results are suppressed.
- Browser playback is sequential, awaits selected output and stops on routing failure rather than leaking onto default speakers. Queue count is bounded; this is not a measured listener-delay budget.
- Home, preparation, authentication, enrollment and live speech now describe the same task/limits. Research engines are under advanced controls; a two-second local microphone/playback check and human output confirmation precede Start. Route quality remains visibly experimental even when models load.
- Enrollment preserves acknowledged uploads across retry, snapshots recording language, releases capture on failures and never reports processing failure as completion. Password/Google login preserve an allowlisted return destination; token console logging removed.
- Private recordings/metadata/QC are restricted; unowned legacy recordings were not relabelled public. No public-demo consent is inferred from a null user id. The private research library is intentionally not available through the shared app. Personal research access/export remains a separate requirement.
- Evaluator includes MT and wall timing. Source WER alone cannot verify a route; fingerprints include actual assets, code, runtime versions/settings and matching references/output hashes, with reviewed STT/meaning/intelligibility/live/policy decisions.
- PRODUCT.md records full-cloning acceptance and provisional LHC scope. README/SDK/current entry points and local report corrected; historical documents remain dated. Hosted Pages was not published.

## Delegation and review

Router query: `router.py find "product designer meeting speech translation onboarding interface implementation"` matched active designer playbook at 0.48. Its lead-generation-specific hosting/booking rules were inapplicable; current project design authority remained DESIGN.md. Requested NIM model `nvidia/nemotron-3.5-super-vl-120b-a12b` returned `designer-nemotron-vl.txt` through the existing free NIM helper, with no paid fallback. Only a sanitized product brief was sent, not code, audio, user records or credentials. Adopted unified prep→check→start→recover→stop direction; rejected incorrect suggestions that no clones exist or enrollment stores no audio. The helper did not expose a resolved-model/usage receipt; report the requested route rather than inventing token counts.

Codex backend worker implemented bounded Python/test changes; Kaggle worker diagnosed/submitted/validated the private replay; independent product reviewer roasted the new frontend. Workers did not delegate. Per-worker token usage unavailable.

Initial roast found five substantive issues in the first repair draft: late sound-check playback after an await, a corrupted blob preview URL, mutable enrollment language, recording failure cleanup and Google return navigation. These were corrected before acceptance. The queue has count protection, not a proven time bound; real continuous playback and audio quality remain the weakest point.

## Verification receipts

- Backend worker: `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTEST_ADDOPTS='--tb=short --show-capture=no' make test` → **68 passed, no skips/deselections**, 27.63s. Original real-model VAD output assertions preserved; harness updated to explicit Start and actual audio pacing. Initial broader run failed three VAD tests, and those failures were resolved rather than hidden.
- `node --test test/session_playback_test.cjs` → **5 passed**: sequential route-before-play, Stop during pending routing, rejected sink without fallback, stale completion after Stop, queue overload.
- Kaggle harness unittest discovery → **5 passed**, including real sample-format normalization and backlog math.
- BrowserOS Neo, owned loopback tab only: home/live examined at 390px and 1440px; horizontal scroll equals viewport; inspected screenshots and corrected hero foreground/background/secondary CTA contrast. This is targeted visual review, not a full WCAG/projector acceptance.
- Browser invalid-JWT test: actionable expired-sign-in state, credential cleared, Start disabled; no model preparation. Browser console read showed expected unavailable-device/no-token warnings without an observed uncaught error in that read.
- Browser isolated in-memory account: actual registration/password login returned to requested enrollment page; authenticated live connection waited for explicit preparation; actual EN→SK models loaded while route remained experimental. No university credentials or owner account were used.
- Browser suspended-device-enumeration probe: mocked mic/recorder, cancelled preparation while enumeration awaited, then resumed it → **0 audio plays**, stopped tracks; no physical microphone captured. The fake APIs were removed by reload.
- Demo asset check: `scripts/demo_preflight.py` passed required local assets; optional OpenVoice converter absent. Asset availability is not a clone/full-route pass.
- JS parse checks and `git diff --check` passed at repair checkpoints. See final appended checks below for the last working-tree state.

## Kaggle T4 result

Private job `yegorby/bp-omnivoice-bounded-session-replay-20261104` v1 is COMPLETE: Tesla T4, pinned model revision/package, 12 measured synthesis rows, nine synthetic schedules, zero job errors. All output WAV hashes/formats/durations and reference normalization were checked. Whole-phrase availability EN 0.649–0.860s, SK 0.691–1.037s, CS 0.558–0.713s; paragraphs 1.403–1.938s. Synthetic Slovak presentation playback delay reached 7.081s and contention 8.081s. These are component/scheduling results, not observed conference delay.

The previous latest failure was reference WAV format validation after successful model load, not a demonstrated import/Python-3.13 defect. The corrected runner converts to mono PCM16/16kHz. Small PCM16 saturation is present; do not claim clipping-free audio. Meaning, pronunciation and identity are human-review-pending. No STT/MT, native streaming, actual meeting transport or actual hardware cancellation was tested in this T4 job. Keep `route_unverified`.

See [Kaggle report](kaggle-report.md) and [sanitized receipt](kaggle-sanitized-receipt.json). Raw outputs/reference corpus remain gitignored and must not be committed.

## Remaining commercial/university gates

1. Human bilingual listening/identity grades for the actual clone outputs; select the exact model/voice route and processing host against that evidence.
2. Continuous complete STT→MT→cloned-TTS→listener rehearsal on intended compute, with p50/p95 speech-end→first-audio, source loss, memory and queue growth; agree limits before measuring. GPU synthesis speed alone does not solve playback duration.
3. Actual LHC room audio/input/return channels, echo isolation, machine/OS, permissions and supervisor acceptance. No more details were available from the owner in this chat.
4. Per-OS clean-machine setup and virtual-device provisioning, physical mic/output/projector/permission/device-loss/reconnect/Stop tests, capacity and restore/support runbook.
5. Deployment-specific privacy/retention and full code/model/voice/dataset licence inventory. Installed `piper-tts 1.3.0` metadata says GPL-3.0-or-later, matching [upstream](https://github.com/OHF-Voice/piper1-gpl); README/registry corrected from MIT. This is a factual correction, not legal clearance or permission to distribute every checkpoint. [OmniVoice upstream](https://huggingface.co/k2-fsa/OmniVoice) locates the API/model; BP's own receipts decide performance.
6. Approve public demo assets separately, and reconcile/publish the separate hosted report only with explicit authorization. No remote git push, deployment or publication occurred.

Roast after fixes: the first-session contracts are substantially safer, but the complete product still lacks human-accepted cloned output and sustained real listener-path proof. Cheapest failure is intelligible but increasingly delayed translated speech. Smallest next experiment is the selected complete route under continuous source audio with an explicit maximum queue-delay budget and bilingual listener grades.

## Final working-tree checks

- Parent reran the expanded offline `make test`; **68 passed, 3 warnings, 28.03s**; see `backend-tests-final.txt` for output. `make test-ui`/Node is a separate optional developer check, not a runtime dependency.
- Real local file-driven WebSocket rehearsal against the isolated loopback test server: translation and TTS audio arrived, and no translated/audio output arrived in the two-second post-Stop observation window. `live-ws-check.txt` records it. This does not independently prove the intended interruption occurred or native inference was cancelled.
- Demo preflight with explicitly overridden `BASE_URL=http://127.0.0.1:8012` passed required assets and live home/capability responses; see `demo-server-check.txt`. This is a loopback HTTP test, not a TLS deployment test or an unmodified `make demo-check` run.
- Final browser preparation check: authenticated real model load succeeded; route remained experimental and Start stayed disabled until sound check/isolation confirmation. Real microphone/headphone confirmation was not performed.
- Detector after UI changes: 21 warnings (12 font preferences conflicting with DESIGN.md; two low-contrast estimates on developer test-STT surface; remaining structural/style heuristics). Main homepage contrast warnings no longer emitted. This is not a full accessibility pass. Raw findings in `design-detector.json`.
- PWA shell cache version bumped and includes the playback controller, so an old shell does not retain the previous lifecycle code. Offline PWA installation/recovery remains untested.
- Current code paths use local-only runtime assets. No local model/package installation occurred. Private T4 job is COMPLETE, not left running. No commit/push/deploy occurred.

Final independent delta review: the product reviewer rechecked all five raised frontend fixes and returned PASS for that bounded source review; no full-product pass is implied. Owned browser tab was closed and both temporary loopback servers exited successfully. Synthetic browser credentials were cleared; the isolated in-memory test database disappeared with its server. No private reference audio was relabelled public or committed. The file-driven speech transcript log is local/ignored.
