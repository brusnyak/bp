# Where the project stands (2026-11-07)

**Verdict:** a single speaker can run live translation on their own machine today (English to Slovak, Czech, German; German validated end to end through the real server on a T4, not yet through the browser UI or on a Mac). The room mode (many listeners, each with an earpiece and their own language) are **not built**. The project is not ready to present as a finished product; it is ready to hand to a technical person as a working, measured base.

## Works today (each line says how it was checked)

| Capability | State | Evidence |
|---|---|---|
| One-command install, default languages | Works in a clean Linux container (setup 352 s, 91 tests pass) | session receipt 2026-11-07; Linux log in `documentation/linux_setup_and_test.md` |
| `--profile mac|nvidia|cpu` and `--langs` | **Code and unit tests only.** Not yet run end to end on a fresh machine of each type | `test/setup_plan_test.py` |
| Real server on a Kaggle T4, EN to DE and EN to SK, both streaming routes | Works with zero server errors. German with agreement every 1.0 s: delay 1.56 s (2 s chunks: 3.89 s). Found and fixed two bugs (German voice not registered, invented stock phrases) | `route_test.json`, `BENCHMARKS.md` section 2b |
| Live route EN to SK on a Mac, MLX turbo, pause mirror | Works; median delay 1.19 s, no errors in a 4-minute browser session | `documentation/evidence/2026-11-07/`, `logs/` |
| Per-language quality and timing | Measured on a T4 for German, French, Polish, Czech, Slovak, English | `LANGUAGES.md`, `BENCHMARKS.md` |
| Slovak accuracy | kinit fine-tune 6.5% on FLEURS, 19.8% on the owner's non-native reading | `sk_compare.json` |
| Guards (loops, caption credits, runaway text) | Unit-tested; one live run | `test/mlx_guard_tests.py` |
| Docs numbers cannot drift | `make check-docs` verifies every marked number against its receipt | `scripts/check_readme_claims.py` |
| Tests | 100+ pass on macOS; CI passes on Ubuntu and Windows; the macOS job had one timing-sensitive failure (`vad_tests.py::test_vad_long_speech_streaming`) | `gh run view 42342557832` |

## Streaming route: validated on a GPU, not yet on a Mac

The accurate route (`BP_INCREMENTAL=1`: decode the growing buffer, commit words two decodes agree on) is validated on a T4 through the real server for English to German and English to Slovak, and is on in the `nvidia` preset. Open items: the real-route transcript error (21-24%) is four times the offline simulation (5%), so voice-activity cuts and chunk edges still cost accuracy; a step below 1.0 s collapses the queue; the Mac cannot keep up with turbo at short steps (0.72 s per call), so it stays on 2 s chunks until Parakeet or a smaller model is validated there. Parakeet is cheap per step but its early guesses are unstable (26-28% error at 0.3 s steps).

## Not built

1. **Room mode**: one speaker, many listeners, each with an earpiece, their own language and either a cloned or a default voice. Design: one recognition pass per speaker; one translation and one voice synthesis **per distinct target language** (not per listener); listeners connect to `ws://HOST/listen?room=ID&lang=de&voice=default` and receive that language's audio and captions; the speaker's session publishes committed fragments to the room. Needs: a room registry in `backend/main.py`, a listener WebSocket, a listener page, per-language model loading, a capacity test (listeners x languages) on Kaggle, and an echo/feedback rule (listeners wear earpieces; the speaker's microphone must not hear them). Size: large (a few days of work).
2. **Voice in your own voice for other people.** Today only a Slovak fine-tune exists (the owner's, local, not distributable). Cloning (OmniVoice, XTTS) needs a GPU and the available weights carry non-commercial licences. Needed: a listener review of identity and meaning on the exact models, and a licence decision.
3. **Fresh-machine installs** for macOS and Windows (profile flags) and the macOS CI flake.
4. **Public site**: `docs/index.html` on the `gh-pages` branch is a Slovak status page dated 3 November with outdated numbers. A new page can be generated from the evidence files; publishing is the owner's decision.
5. **Licences (owner decision 2026-11-07):** the project is a configuration layer. It bundles no model weights (checked: no `.onnx`, `.bin`, `.pt` or `.safetensors` file is tracked) and its setup downloads each model from its publisher, so the user accepts that model's licence. The list is in `LICENSE_INVENTORY_2026-11-06.md`. One boundary to keep in mind: whoever runs it as a shared or hosted service (a room for other people) is the operator and should pick models whose licences allow that (Piper is GPL, OmniVoice and XTTS weights are non-commercial).
6. **Operations for shared use**: capacity limits per GPU, retention, monitoring, a support owner.

## Decisions needed

- Build room mode now that the streaming route is validated on a T4? (the LHC scenario; the larger piece)
- Publish a regenerated `docs/index.html` to the project site?
- Which cloned-voice route to pursue for listeners?
