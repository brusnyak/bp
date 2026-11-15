# Evidence notes for models/evidence.json (checked 2026-11-15)

Basis: every number in evidence.json was read from the cited file. Profiles beyond Tesla T4 and M1 Pro 16 GB are spec extrapolations (curated min_ram/min_vram/devices); only `recommended` cells have a receipt or a preset behind them.

## Gaps (no measurement for the model)
- 18 of 40 ids have no receipt: lessac, kokoro, piper-voice-clone-en/-en-v2/-sk, piper-sk personal, piper-generic-sk-omni, whisper-tiny, madlad, chatterbox, qwen3-tts, webrtcvad, silero, hybrid-tts, qwen3-omni, seamless, rvc-mlx, rvc-onnx.
- Kokoro "RTF 0.03, 1.33 GB on T4" is declared in LICENSE_INVENTORY_2026-11-06.md ("load test pending"); no receipt under documentation/evidence.
- No receipt on any 8 GB GPU, any 24 GB+ GPU, any x86 laptop CPU or any server CPU. Those columns are spec-only.
- Piper lessac RTF "~0.04" is from fundamentals_deep.md prose; the only receipt is the sibling voice en_US-john-medium (RTF 0.03).
- curated.json has 41 entries but 40 ids: `piper-sk_SK-lili-medium` appears twice (rows 2 and 33, different `why`).

## Conflicts between docs
- Higgs VRAM: curated 24 GB; TTS_LANDSCAPE says ~10 GB bf16; PLAN_2026-11-10 says >=6 GB; receipt peaks 13.7-13.9 GB on T4 fp16 with WER 100%.
- Nemotron Slovak: curated lists `sk`; PLAN says WER 0.75-0.86 (2026-10-01 bench) and 28.2%; mac-nemotron.env says 41%. Receipt for 41% not found. Curated `devices: ["cpu"]` but it is MLX/Apple only.
- Parakeet: curated "~6.3% EN WER"; lang_matrix gives EN 5.1%, DE 6.3% (the 6.3 is German). Parakeet "slow" T4 speed in lang_matrix ran on CPU; e2e_chain has the CUDA number (0.11 s/clip).
- OmniVoice RTF: 0.30 (probe_v2, 12 steps), 0.23-0.38 (load_tests s5), 0.18-0.34 (STATUS), 0.40-0.65 (load_tests header), 0.42-0.45 (TTS_LANDSCAPE). MPS: STATUS 0.8-1, fundamentals "~1.0", fundamentals_deep "2.9->2.5".
- OmniVoice "5.8 GB at 12 steps": load_tests s5 says 5.7-5.9 GB for the whole cascade (whisper+MT+OmniVoice), not OmniVoice alone.
- X-Voice: TTS_LANDSCAPE "T4 RTF 0.61-0.80 (measured)" predates the receipt; kaggle_xvoice_t4 gives 0.37-0.47 (NFE16, stage 1) and 0.80-0.89 (NFE32). Curated "2.2-2.7 s" is short phrases only (long: 5.8-6.9 s). TTS_LANDSCAPE: Mac MPS RTF 2.9-4.2, curated lists mps/cpu as devices.
- Chatterbox: curated languages include sk/cs; TTS_LANDSCAPE s1.8: 23 languages, no SK/CZ. LICENSE_INVENTORY calls it EN-only.
- Qwen3-TTS licence: LICENSE_INVENTORY MIT vs curated/TTS_LANDSCAPE Apache-2.0.
- Mac default: STATUS (2026-11-08) says mac.env lag 3.0-3.6 s; PLAN (2026-11-10) says Nemotron 0.5-0.6 s but mac.env was not changed (setup does not install mlx-audio). Default is still Whisper turbo.
- LANGUAGES.md "CPU only: German about 11%" is the T4 whisper-small number, not a CPU receipt (presets/cpu.env says so).

## Home page numbers vs receipts (ui/home/home.html lines 83, 154-162)
- "0.6 s Default, Mac M1" and "0.6-0.8 s to first sound": matches Nemotron EN->SK on generated English (0.621 s adaptive), but the default preset is Whisper turbo (3.0-3.6 s, STATUS; README paced replay 0.82-1.19 s). Real speech with Nemotron: 1.098 s median (route_nemotron_en_sk_real_speech_mac.json).
- "0.8 s Longer segments" and "1.0 s Word by word": no receipt located. 2026-11-15 sweep D_wait05 is 0.792 s median, p90 1.387 s.
- "0.5 s Default, Tesla T4": no receipt found. T4 route runs: 1.56 s (EN->DE agree 1.0 s), 1.89-1.97 s (EN->SK), 3.89 s (2 s chunks), CS->EN p50 3.48 s (route_receipt.json).
- Caption "STT 0.40, MT 0.07, TTS 0.13": local-paced-n1.json shows stt_time 0.38-0.44 s, mt_time 0.085-0.10 s, tts_time 0.136 s. MT is 0.085-0.10, not 0.07.
- Caption cites documentation/implementation-2026-11-04/, which README marks as archived; AGENTS.md says dated archives are not acceptance evidence.
- Models page note: Higgs, XTTS and X-Voice numbers in curated `why` match receipts (XTTS first audio 0.44-0.53, RTF 0.52-0.64, 1.86-2.49 GB peak; Higgs WER 100%).
