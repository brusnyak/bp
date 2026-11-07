# Where the project stands (2026-11-07)

**Verdict:** a single speaker can run live translation on their own machine today (English to Slovak, Czech, German; German measured per component, not yet run live in the browser). The accurate low-error streaming route and the room mode (many listeners, each with an earpiece and their own language) are **not built**. The project is not ready to present as a finished product; it is ready to hand to a technical person as a working, measured base.

## Works today (each line says how it was checked)

| Capability | State | Evidence |
|---|---|---|
| One-command install, default languages | Works in a clean Linux container (setup 352 s, 91 tests pass) | session receipt 2026-11-07; Linux log in `documentation/linux_setup_and_test.md` |
| `--profile mac|nvidia|cpu` and `--langs` | **Code and unit tests only.** Not yet run end to end on a fresh machine of each type | `test/setup_plan_test.py` |
| Live route EN to SK on a Mac, MLX turbo, pause mirror | Works; median delay 1.19 s, no errors in a 4-minute browser session | `documentation/evidence/2026-11-07/`, `logs/` |
| Per-language quality and timing | Measured on a T4 for German, French, Polish, Czech, Slovak, English | `LANGUAGES.md`, `BENCHMARKS.md` |
| Slovak accuracy | kinit fine-tune 6.5% on FLEURS, 19.8% on the owner's non-native reading | `sk_compare.json` |
| Guards (loops, caption credits, runaway text) | Unit-tested; one live run | `test/mlx_guard_tests.py` |
| Docs numbers cannot drift | `make check-docs` verifies every marked number against its receipt | `scripts/check_readme_claims.py` |
| Tests | 100+ pass on macOS; CI passes on Ubuntu and Windows; the macOS job had one timing-sensitive failure (`vad_tests.py::test_vad_long_speech_streaming`) | `gh run view 42342557832` |

## Measured but not built

- **Accurate streaming route.** The simulation shows that decoding the growing buffer every 1.0 s and committing only words two decodes agree on gives 5.0% English and 11.2% German word error at about 2.3 s, against 19% and 33% for the current 2 s chunks at about 1.2 s. The old `BP_INCREMENTAL=1` code has the building blocks (agreement, fragments) but duplicated and dropped fragments in the one real-route test with Parakeet. It needs a real-route test on a GPU box before it becomes the default. Next action: a Kaggle kernel that runs the actual server and the paced replay with this route (staging the code as a dataset).
- **Parakeet for low-latency captions.** Cheap per step (0.1-0.3 real-time factor) but unstable early guesses (26-28% error at 0.3 s steps).

## Not built

1. **Room mode**: one speaker, many listeners, each with an earpiece, their own language and either a cloned or a default voice. Design: one recognition pass per speaker; one translation and one voice synthesis **per distinct target language** (not per listener); listeners connect to `ws://HOST/listen?room=ID&lang=de&voice=default` and receive that language's audio and captions; the speaker's session publishes committed fragments to the room. Needs: a room registry in `backend/main.py`, a listener WebSocket, a listener page, per-language model loading, a capacity test (listeners x languages) on Kaggle, and an echo/feedback rule (listeners wear earpieces; the speaker's microphone must not hear them). Size: large (a few days of work).
2. **Voice in your own voice for other people.** Today only a Slovak fine-tune exists (the owner's, local, not distributable). Cloning (OmniVoice, XTTS) needs a GPU and the available weights carry non-commercial licences. Needed: a listener review of identity and meaning on the exact models, and a licence decision.
3. **Fresh-machine installs** for macOS and Windows (profile flags) and the macOS CI flake.
4. **Public site**: `docs/index.html` on the `gh-pages` branch is a Slovak status page dated 3 November with outdated numbers. A new page can be generated from the evidence files; publishing is the owner's decision.
5. **Licence audit** for distribution: Piper is GPL-3.0-or-later (not bundled), Parakeet and Opus-MT need attribution, personal fine-tunes and OmniVoice are restricted. `documentation/LICENSE_INVENTORY_2026-11-06.md` lists them; no legal review was done.
6. **Operations for shared use**: capacity limits per GPU, retention, monitoring, a support owner.

## Decisions needed

- Build the accurate streaming route and test it on the real server on a T4? (recommended first)
- Then build room mode? (the LHC scenario; the larger piece)
- Publish a regenerated `docs/index.html` to the project site?
- Which cloned-voice route to pursue for listeners, given the licences?
