# Voice similarity QC — 2026-08-19

Objective speaker-embedding cosine similarity, not RTF, not subjective description. Tool: `resemblyzer` 0.1.4 (`VoiceEncoder`, GE2E speaker-encoder architecture, `github.com/resemble-ai/Resemblyzer`), run via `scripts/voice_similarity_qc.py` in an isolated venv (`resemblyzer`, `piper-tts`, `setuptools<81` — see script header).

## Reference audio

`speaker_voices/voice_rec_1m.wav` — 41.7s, user reading the Rainbow Passage. Same file used as the XTTS speaker reference for both the DE and CZ bootstraps.

## Raw cosine similarities vs. reference

| Sample | Source | Cosine similarity |
|---|---|---|
| `piper_personal_en` | EN test sentence synthesized live via the fine-tuned `en_US-personal-medium.onnx` Piper voice (`PiperVoice.load` + `synthesize_wav`) | **0.7474** |
| `xtts_de` | `/Users/yegor/Desktop/personal_voice_test_de_xtts_not_piper.wav` — XTTS v2 cross-lingual clone, German | **0.8406** |
| `xtts_cz` | `/Users/yegor/Desktop/personal_voice_test_cz_xtts.wav` — XTTS v2 cross-lingual clone, Czech (this session's bootstrap) | **0.8309** |

## Threshold reference point

Resemblyzer's own `demo05_fake_speech_detection.py` draws its real/fake decision line at cosine similarity **0.84** (`plt.axhline(0.84, ls="dashed", label="Prediction threshold", ...)`). That's the only numeric threshold published anywhere in the tool's own repo/docs — confirmed by fetching the script directly, not assumed. It's illustrative for one fake-speech-detection demo (same reference speaker, real vs. TTS audio), not a formally calibrated general speaker-verification cutoff. No number is given in resemblyzer's PyPI README beyond "reject similarity scores below a threshold" (no value stated).

No editorializing beyond that: the three raw numbers above are 0.7474, 0.8406, 0.8309.

## Note on the DE sample file

`/Users/yegor/Desktop/personal_voice_test_de_xtts_not_piper.wav` was confirmed present (209,484 bytes) early in this session, then found missing partway through (`ls: No such file or directory`) — not removed by any command run in this session/worktree. Restored from the git-committed original bytes (`git show 80fa192:bootstrap_de/de_000.wav`, same 209,484-byte file) before running the QC script, so the DE number above is the genuine original XTTS output, not a re-synthesis.
