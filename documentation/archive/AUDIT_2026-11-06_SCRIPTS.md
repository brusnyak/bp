# Scripts, evaluation, packs and specifications audit — 2026-11-06

Verdict: REWORK before self-service public release. New validators correctly distinguish smoke tests from reviewed quality, but advertised workflows still include broken adapters, wrong references and misleading latency metrics. Preserve receipts and fix the evidence boundary before further model tuning.

Scope: first-party text in scripts/, bench/ code and reference text, languages/, specs/; bp/ compared against backend/app counterparts. Exclude raw benchmark result JSON (root auditor owns those), model/audio binaries, piper_models sidecars and third-party whisper_streaming_vendor code. Read all scoped sources in batches. Exact copies use byte equality plus independently reviewed original; near-copies use complete baseline and full diff. No GPU/model execution, install, download, remote mutation, deletion or implementation edit. Existing dirty/untracked content preserved.

## Fresh checks

- `.venv/bin/python scripts/bp.py model-harness --stage stt --model base --device cpu`: exit 2, `unrecognized arguments: --cmd model-harness`, before inference.
- `.venv/bin/python -m unittest discover -s scripts/gpu_bench/kaggle_canary -p test_validate.py -v`: 6/6 pass.
- `.venv/bin/python -m unittest discover -s scripts/gpu_bench/kaggle_session -p test_replay.py -v`: 6/6 pass, includes normalization.
- `.venv/bin/python bench/metrics.py`: metrics self-test PASS.
- `python3 scripts/ui_manifest.py --check`: both manifests ok.
- Stdlib ast.parse / json.loads on scoped Python/JSON/notebook sources: no syntax/JSON errors. No application imports or bytecode writes.
- SHA-256 byte comparison and complete diff reviewed bp mirrors, four direction-job copies and dataset script copies. `git status --short bp scripts bench languages specs` identifies bp/ as untracked.

## Findings and smallest corrections

### P1: advertised model-harness CLI and adapters are broken

`scripts/bp.py:621` forwards dispatch-only cmd as `--cmd model-harness`; reproduced parser failure above. Exclude that field and add a no-model parser regression. `scripts/model_harness/run.py:73` always constructs FasterWhisperSTT, so its README's Canary example is interpreted as a Whisper model. `:111` only uses legacy TTS_ENGINES; the README's Kokoro adapter is absent. `:94` turns opus-mt into ct2_models/opus-mt, unlike the real canonical directory. `:37` provides short prefix references while scoring whole recordings: invalid whole-clip WER. `:58` includes raw model paths in output names, allowing nonexistent nested paths. Output `:157` overwrites a stable filename, contrary to never-overwrite README. `:140` checks declared resource text, not observed VRAM; unknown declarations pass. TTS records requested device without selecting it.

Fix: only expose implemented registry adapters, require complete fixture/reference pairs and input hashes, sanitize IDs, append unique run IDs, record actual device and observed memory separately from declarations. No existing generic harness score should establish acceptance.

### P1: old route runner's p50/p95 is session elapsed time, not latency

`scripts/gpu_bench/kaggle_full_route/kaggle_full_route.py:135` collects absolute emission timestamps since replay start and quantiles them as first_audio latency. It calls StreamingPipeline directly and awaits inference inside the sending loop. Quarantine those fields as historical direct-pipeline diagnostics. The newer separate job.py uses production onset metrics and warmup+3 repeats, a stronger basis after the fixture/provenance issues below are fixed.

### P1: wrong-language route fixtures and offline/provenance drift

`scripts/gpu_bench/kaggle_de-en/job.py:62` and kaggle_fr-en/job.py:62 use an English recording as a German/French placeholder. All five copied route runners still store audio.language='cs' at :377 and CS/EN-labelled field names. The base full_route/job.py uses a Czech default clip regardless of changed source args; --audio help promises the pack's clip without implementing that lookup. Reject mismatched fixture language. These runs cannot validate German/French source support.

`scripts/gpu_bench/kaggle_full_route/job.py:152` clears offline flags and downloads/converts MT despite its pre-provisioned offline docstring. `:158` converts weights without tokenizer save. Fingerprint occurs before runtime conversion and covers code, not effective weights/tokenizer. Use one parameterized runner, explicit setup, pinned/hash-verified assets and final effective provenance. Performance measurement alone does not verify translation meaning.

### P1: engine gates/selftest overstate quality

`scripts/bp.py:287` declares sim_min but Piper gate only tests RTF. `:395`–`:403` turns one silence frame into false-start rate; speech-detection result is unused. Unknown threshold combinations return success with a warning. `:347` translates twice per sentence, inflating latency. Selftest uses silence STT, independent fixed MT text, Czech Piper at :462 for Slovak synthesis at :475. This is component smoke, not EN→SK quality acceptance.

Fix: rename smoke checks, reject absent gate criteria, single MT call, target-language voice, separate STT WER, MT meaning, TTS intelligibility/identity, onset latency, completeness and cancellation checks.

### P1: pack evaluator bypasses effective production settings

`scripts/language_packs.py:361` constructs FasterWhisperSTT using only model, omitting engine choice, prompt, beam/profile; :377 always directly constructs Piper. Whole-clip wall is not live onset latency. Preserve :190's improved fail-closed full-route quality-review and fingerprint checks. Test writer always records experimental and no review/promotion command completes the self-service workflow. resolve_stt treats remote supported sizes as resolved without installed cache. hf_model_info treats arbitrary nonempty terms as commercial-ok except recognized NC; local MT folders are likewise assumed commercial-ok.

Fix: evaluate the effective production route/config, distinct installed/resolved/smoke/reviewed states, explicit reviewed promotion artifact and structured licence provenance. Current GPL/model/voice licence assessment remains the root audit's responsibility.

### P1: setup is hardcoded broad bundle, not chosen-route clean install

`scripts/setup.py:22` hardcodes three voices; scripts/convert_models.py:17 converts eleven MT directions, ignoring chosen packs; setup MT_PAIRS is unused. `convert_models.py:30` skips directories on model.bin alone, so tokenizer failure remains permanently partial. finally deletes fixed .venv-convert even if pre-existing. Downloads/conversions lack revision/hash pinning. Setup installs certificate, Voice Lab grade-store and smoke selftest but has no supported CPU/CUDA OmniVoice extras profile.

Fix: manifest-driven minimal baseline plus optional profiles, asset completeness/integrity/licence checks, owned temporary converter directory; fresh three-OS acceptance without owner-local assets/cache. Do not run setup during a read-only audit.

### P1: old clients fail open or use obsolete protocol

`scripts/live/paced_replay.py:127` records server errors without failing, permits no voiced rows, cancels receiver without awaiting/propagating failures. Source-end uses transcript arrival minus STT time; last-transcript association may misattribute concurrent audio. summarize_replay.py fails on empty voiced output. Use production onset timestamps and assert expected text/audio completeness.

`scripts/vad_close_probe.py:41` passes TLS context to default ws://, lacks authentication, sends config instead of config_update and waits for translation instead of translation_result; 60ms audio chunks sleep20ms, a 3x media pace. meeting_live.py invokes missing scripts/live_direction_probe.py; pipeline_latency_probe.py invokes missing scripts/stt_parakeet_spike.py and legacy clip names. Consolidate one authenticated production replay client; fail on error, empty, mismatched or incomplete output.

### P1: simulated figures can be mistaken for measured live timing

`scripts/meeting_sim.py:119` constructs playback from assumed0.15s STT tail plus first-chunk MT/TTS costs, omitting measured STT duration; output can overlap without sequential player constraints. Derived streaming boolean proves a simulation, not live streaming. build_demo_charts.py:127 divides total batch STT time by only the capped seven shown rows, inflating per-clip means; its Gantt legend hardcodes ~0.1/~0.4s claims and draws assumed streaming spans.

`scripts/archive/generate_charts.py:89` hardcodes Piper baselines, :99 STT, :239 RTF estimate, :282 subjective artifact scores ending0 without grading receipt. archive/test_full_pipeline.py:169 substitutes expected text for empty STT. Keep historical evidence intact; quarantine derived figures from current claims. Require typed receipt scope for charts.

### P1: current tuning script is incompatible with current Cascade

`scripts/gpu_bench/tts_tune.py:34` constructs load_bench.Cascade(a), whose current constructor at load_bench.py:50 reads a.steps and a.batch; tuning parser defines neither. Its identical kaggle_data/tts_tune.py runs with the older neighboring Cascade, which lacks those fields. Unify versioned harness imports, add lightweight argument-contract test before an expensive remote run.

### P2: audio-routing provisioning has clean-start blockers

`scripts/audio/setup_windows.ps1:117` names a random temp directory but never creates it before :122 download. Documentation invokes powershell, yet ?? syntax requires PS7, incompatible with default Windows PowerShell5.1. Linux setup_linux.py:32 module IDs remain None because enumeration never populates them, so :160–:165 removal is a no-op returning success. :150 loses original successful module ID on remap failure, breaking rollback; existing sink without remap is not repaired. macOS parser strips indentation before detecting headings and only handles1/2 channels. No actual driver/routing changes made.

Fix fixture-test parsing/module lifecycle, then verify real capture→translation→virtual-mic round trip on each supported OS. Device-name presence alone is insufficient.

### P2: bp/ duplicate runtime materially diverges

This worker's explicit scope is 23 Python +1 auto_config.json: 17 identical +6 different Python, plus identical config =18 identical/24 total. Broader backend audit may include additional mirrored files/sidecars/vendor files, so its denominator differs. Hash ledger gives the exact paths. Six differing copies: main, pipeline, registry, streaming/pipeline, stt/faster_whisper_stt, tts/base.

bp main retains disabled mid-speech chunks while root currently processes them, personal-SK remapping/debug audio and different no-speech threshold. bp streaming initializes most components outside its lock and bypasses the root's TTS licence gate. bp Pipeline has a second pool; release updates last_used before TTL comparison, preventing expiry there. bp registry strictly rejects unknown targets while root falls back to English. CWD/import path can select different behavior. Classify snapshot vs package ownership first; removal/relocation requires explicit owner-approved cleanup, no blind copy normalization.

### P2: stale packs/specs and corpus tooling

`languages/packs/pl-en.json:20` uses English clip/reference for Polish STT; DE→EN notes ask for German TTS although targetEN; EN→DE notes name another voice. Null fixtures remain scaffolds. 003 tasks T012 claims pack-driven setup, T006 retains VERIFIED historical full-clip tests conflicting with new acceptance; T030 says DONE under open checkbox, duplicate tasks/phases remain. plan links moved scripts/eval_stt.py and test/interrupt_smoke_test.py and old test counts; spec calls0.7/1.4s real-time without timing origin. 001/002 stay Draft at original paths while only _archive/README says historical.

prepare_voice_corpus writes temp files even in dry-run; equal stems across audio extensions collide; heuristic energy dedupe is not identity proof. bulk_hq adds Slovak NEW passages even for --lang en. make_reading_script overwrites proofread SK material with MT. machine_listen_qc aligns audio thirds to word-count thirds without timing alignment. These diagnostics must not promote language/voice quality automatically.

## Cleanup and commercial self-service acceptance

1. Evidence maintainer: fix CLI/adapters/fixtures/metrics and consolidate runner/client schemas. Acceptance: no-model parser/scoring tests, mismatch/empty/error cases fail, complete expected output tracked, no fake references/placeholders accepted.
2. Release maintainer: one package/import tree, chosen-route CPU/CUDA profiles, immutable model/voice manifests. Acceptance: fresh macOS/Windows/Linux start without owner audio, models, database, caches or credentials, then authenticated public-fixture replay. Optional cloned GPU route separately installs/tests.
3. Documentation maintainer: one active requirements ledger mapping requirement→implementation→receipt→gap, source-aware index and archive banners. Acceptance: supported commands/links work; all quality/speed claims identify scope/device/repeats/reference and reviewed outputs. Preserve raw receipts/personal models; propose deletes separately.

Best achievable version: a new user selects SK→EN or CS→EN, installs an explicit CPU or GPU profile, sees honest support/quality status, optionally enrolls an authorized voice, and runs public fixtures through the UI's actual route. Gap: owner-only fixtures, broken/bypassed eval entrypoints, divergent packaging, missing quality-promotion workflow. Commercial availability additionally needs root licence/security/runtime findings resolved.

suggest: create a public consented SK/CS/EN fixture pack with reviewed English translations and names/numbers/negation checks before more speed work; this closes the largest gap between fast audio and demonstrated meaning preservation.

## Per-file coverage ledger

R=complete source/text/config read. E=byte-identical equivalence to reviewed original. D=complete baseline plus complete diff reviewed. For bp runtime originals, backend auditor owns full baseline review; this worker owns equivalence and divergent hunks. Hashes bind the audit to actual working-tree bytes, not a clean commit. Raw bench result JSON belongs to the root benchmark auditor.

| File | Lines | SHA-256 | Coverage / assessment |
|---|---:|---|---|
| `bench/README.md` | 35 | `3eb3c685983f19d4dc0ee6b47f24efaec90aa81684cc17f077c3be5fe14a8ae4` | R: Honest single-speaker/duplicate-reference/STT-only limits; frozen live differs runtime. |
| `bench/RESULTS.md` | 21 | `566a56f16c53eb26d37b6cc57232bf0b4be300edb5b9c4009670e0f5a2cc1fd9` | R: Generated subset table; raw receipt/table consistency root-owned. |
| `bench/make_table.py` | 11 | `cac084020c6dfcfdb6f84a644fb73bbbf3a181c3ade3bded402dbd062720f6fa` | R: Overwrites generated table; hardcodedCPUint8 label/no schema validation. |
| `bench/metrics.py` | 41 | `2eb6f8f82b659fd0f065dbd1d148bed21ef9ff00a0a91760074ddc4ff17a53e4` | R: Reference-normalized edit distance, diacritics preserved; selftest passes; emptyref divides0. |
| `bench/refs/me_sk_script.txt` | 1 | `1c016f4c39d0a1d98ce53448ec849b46779ef6fe97bfa547b0cf5dc07700735e` | R: Slovak reference read, audio alignment unreviewed; trhove and_b are identical text, not independent material. |
| `bench/refs/me_sk_trhove.txt` | 1 | `9db98dcc6352100496433d72442a9cdb13f11c75590d186d28cb2acd43fb8ff5` | R: Slovak reference read, audio alignment unreviewed; trhove and_b are identical text, not independent material. |
| `bench/refs/me_sk_trhove_b.txt` | 1 | `9db98dcc6352100496433d72442a9cdb13f11c75590d186d28cb2acd43fb8ff5` | R: Slovak reference read, audio alignment unreviewed; trhove and_b are identical text, not independent material. |
| `bench/run_stt.py` | 109 | `a496ed949f72cceb85e0d62644ae9a7fc4401999de3df70d9d360cfe38fafb11` | R: FrozenCPUlocal-onlyVAD copy, remainder discarded at close; STT-onlyRTF, append runs. |
| `bench/run_stt_vad.py` | 87 | `6e20b82fbff27dfbd961d1137968358072488131f6dab0f82778a8f982a49dee` | R: VariantASR diagnostics; weaker provenance than base, output dir not created. |
| `bp/app.py` | 589 | `e35095eb037f94308ef121bb4303bf0dfbeb9d2a2b5f7e618a0bdce990653872` | E: byte-identical app.py; backend auditor owns baseline. |
| `bp/backend/config/auto_config.json` | 46 | `d3078d3fb09436fba143384dbcf0ecfe62725b255aebe5ad4638414a99bc7919` | E: byte-identical backend/config/auto_config.json; backend auditor owns baseline. |
| `bp/backend/hardware.py` | 87 | `1399a075d54f267a4c75916c413c54fedf0596f7f4b3e986c75a23822f9972c7` | E: byte-identical backend/hardware.py; backend auditor owns baseline. |
| `bp/backend/hardware_scan.py` | 335 | `e7eb26f4328dfbf7bce478f87bdcfadbe5f2aa4a21802ff3c76db61966fa4671` | E: byte-identical backend/hardware_scan.py; backend auditor owns baseline. |
| `bp/backend/latency_profiles.py` | 14 | `81cecf190437b2b96f2367468a34f61e33d399562778544b1b0a5cddc3331254` | E: byte-identical backend/latency_profiles.py; backend auditor owns baseline. |
| `bp/backend/main.py` | 2562 | `4306fece42486aac6a8fcee8bd0f2c47bb899023090ef56caf59ad1db7e46ecf` | D: full differing hunks against backend/main.py reviewed; divergent runtime above. |
| `bp/backend/mt/convert_opus_mt_to_ct2.py` | 61 | `91771fd5ca1e3efb4672fb52859da7cd8234868a963007f56487f1b58281e56a` | E: byte-identical backend/mt/convert_opus_mt_to_ct2.py; backend auditor owns baseline. |
| `bp/backend/mt/ctranslate2_mt.py` | 230 | `fa484faa965e9ec4de01f7ad6ade4d65737e1ba0632cc7e5d1b9fa267c886bd1` | E: byte-identical backend/mt/ctranslate2_mt.py; backend auditor owns baseline. |
| `bp/backend/mt/hybrid_mt.py` | 134 | `4c7342ffbd4d38088b6749421da60521ff088dba70fcd900886b6f9091fb7c5d` | E: byte-identical backend/mt/hybrid_mt.py; backend auditor owns baseline. |
| `bp/backend/mt/nllb_mt.py` | 150 | `f7611b17410874348840c5d86dabd9b1f8501c458037a45b612d31a2e7107ba2` | E: byte-identical backend/mt/nllb_mt.py; backend auditor owns baseline. |
| `bp/backend/mt/ssbd_ctranslate2_mt.py` | 134 | `86d56415144259559637c10e8930d7a12054a85c096c2ee33dbe243dd80aad5d` | E: byte-identical backend/mt/ssbd_ctranslate2_mt.py; backend auditor owns baseline. |
| `bp/backend/pipeline.py` | 373 | `a75c3a08d79f19260eb26f2c707cc952fae3ed311e84b8921300feea62825306` | D: full differing hunks against backend/pipeline.py reviewed; divergent runtime above. |
| `bp/backend/registry.py` | 409 | `bef94b665d9704e5f2290bf4550cb3433d2089802d76a8720f71e643100ffe17` | D: full differing hunks against backend/registry.py reviewed; divergent runtime above. |
| `bp/backend/session_manager.py` | 235 | `f28575ec4796911e65f33f79d32a53d9d17ed99cfa168cc74a352c0b8b4374cc` | E: byte-identical backend/session_manager.py; backend auditor owns baseline. |
| `bp/backend/streaming/__init__.py` | 10 | `2ddb5d58a53bcbe8fedf9e75a06103af1394a5c3a01ec87a5d6e441b58cefede` | E: byte-identical backend/streaming/__init__.py; backend auditor owns baseline. |
| `bp/backend/streaming/pipeline.py` | 343 | `b3767cbb5c16e74afb3858144222f6f27d26a509ae33e5cd1fc9a54e192ab127` | D: full differing hunks against backend/streaming/pipeline.py reviewed; divergent runtime above. |
| `bp/backend/stt/faster_whisper_stt.py` | 107 | `e3fb90954e17c20642120297325d11ec684978957c3fb80e30657d8a64c2e46c` | D: full differing hunks against backend/stt/faster_whisper_stt.py reviewed; divergent runtime above. |
| `bp/backend/stt/incremental.py` | 64 | `27e9bf7e9ca483613f7111abff2e0879060af46309bd8ad15c8e83cf82cf5353` | E: byte-identical backend/stt/incremental.py; backend auditor owns baseline. |
| `bp/backend/stt/streaming_captions.py` | 69 | `ea16d2049319028003a632b97abdee4c105c3a8877084735b2205cb9163816d7` | E: byte-identical backend/stt/streaming_captions.py; backend auditor owns baseline. |
| `bp/backend/tts/base.py` | 132 | `d09c3bd42901a6add3d2c77f4540508bf3eb39e569483b19788d9d0fe6154f45` | D: full differing hunks against backend/tts/base.py reviewed; divergent runtime above. |
| `bp/backend/tts/coqui_tts.py` | 475 | `3e26e9569f17c199fdd3f1b9da43e6c9ef6b460a7af832647a065855a9689453` | E: byte-identical backend/tts/coqui_tts.py; backend auditor owns baseline. |
| `bp/backend/tts/download_piper_models.py` | 78 | `8c9c2cadf53771f209907c1498eada8630d285721cd0d5e8c763ec67fb7ce0db` | E: byte-identical backend/tts/download_piper_models.py; backend auditor owns baseline. |
| `bp/backend/tts/hybrid_tts.py` | 212 | `c5a645bf66b188da6dddf175c327c6e99d096beb6f6e398096b5b77a98098177` | E: byte-identical backend/tts/hybrid_tts.py; backend auditor owns baseline. |
| `bp/backend/tts/omni_tts.py` | 105 | `cd2888f4e1efd6df5627ee8ad36e5b3fa249cd669bc03517d605240d96655a58` | E: byte-identical backend/tts/omni_tts.py; backend auditor owns baseline. |
| `languages/README.md` | 38 | `c57d6c07544a2056286a55e286b41dd665fad306429529373b11a03287dc9862` | R: supported/test/add semantics stale vs full-route reviewed fingerprint and scaffold-only add. |
| `languages/packs/cs-en.json` | 43 | `9a23f7a1bfecbbdd9c2604d7adc3c57bd22f89d724a194bf36d007458de1c57d` | R: CSbase/ENvoice, low_latency diagnostic profile, quality acceptance pending. |
| `languages/packs/de-en.json` | 13 | `46a2f789365fbed221bc3ee4f39c1c3daa3cf436fe49b872f05e2d3899111e8e` | R: English target correct; notes wrongly requestDEvoice; no fixture. |
| `languages/packs/en-cs.json` | 27 | `4eaa6f583d0bf33fe3e44f064b75ecc64bb9618dc5ce49d7f1ebca9bb61a5232` | R: Declared STT/MT/PiperTarget/VAD inspected; English clip exists, reference null. |
| `languages/packs/en-de.json` | 13 | `83614e95214883053b76b8a652f62c1a639e868eece9b5458b18bf705fb5191a` | R: Declared karlsson-low conflicts notes karl-medium; no fixture. |
| `languages/packs/en-fr.json` | 13 | `37dc9218d9d7e137b112457df7fc6fbf0838b88441356787d0cce17e4dc8958e` | R: Declared STT/MT/PiperTarget/VAD inspected; null clip/reference, scaffold only. |
| `languages/packs/en-hi.json` | 27 | `e3e598f53d655267ff5936d804c1390cf02ffe96d7d54e787e52006ab0a0d188` | R: Declared STT/MT/PiperTarget/VAD inspected; null clip/reference, scaffold only. |
| `languages/packs/en-pl.json` | 27 | `a27ca539f884fb53406f4a05da20fabc91f0deeb919ae8b496e021d515ad681e` | R: Declared STT/MT/PiperTarget/VAD inspected; null clip/reference, scaffold only. |
| `languages/packs/en-sk.json` | 28 | `459efa64048c9e733aa5dc1b7862282a6cd9737c49fba64ed2f6073ab72d905d` | R: PersonalSK with generic fallback; explicit unverified personal-model provenance. |
| `languages/packs/en-zh.json` | 27 | `2e8384a14f3a052f38de2ec60f8a352a23e15465bd131f5f781558839daff459` | R: Declared STT/MT/PiperTarget/VAD inspected; null clip/reference, scaffold only. |
| `languages/packs/fr-en.json` | 13 | `3a99ecf028125632a1c593a4089b86449132cbbe21739a6f68c65fc108a8427c` | R: Declared STT/MT/PiperTarget/VAD inspected; null clip/reference, scaffold only. |
| `languages/packs/hi-en.json` | 27 | `406639bf9356ee870875cc8d10d0aaa3e693537c0ec2f310225ed5c694f34aa5` | R: Declared STT/MT/PiperTarget/VAD inspected; null clip/reference, scaffold only. |
| `languages/packs/ja-en.json` | 27 | `4774067367769fd9ed0924e698ea37499e9f6ac1ab017f3ff0a7c21a208424c2` | R: Declared STT/MT/PiperTarget/VAD inspected; null clip/reference, scaffold only. |
| `languages/packs/pl-en.json` | 27 | `29d0da3a44df0c53a0acd592a0cfa7eceda7cadaa6dcd7e0ebd009c804517fe4` | R: English clip and prefix reference mismatch Polish source; invalid evaluation fixture. |
| `languages/packs/sk-en.json` | 22 | `8d04eefd0cc78d0d379e1b34a4b712c21c0708de8d5afca2a7d34ed4ad436b1a` | R: small-sk/prompt/ENvoice; evaluator drops prompt, source quality experimental. |
| `languages/packs/zh-en.json` | 27 | `fdd63fe31aa1042e4b1806aa4de23cfa50455c287c3059aec731b4e6133984aa` | R: Declared STT/MT/PiperTarget/VAD inspected; null clip/reference, scaffold only. |
| `scripts/archive/benchmark_full_pipeline.py` | 197 | `e41b7f8eb158f869e28770305c25e36a0bc8c95186b9fcac7312c632ad011fc9` | R: Historical random-noiseSTT+unrelated component means, invalid end-to-end target claim; stale import root. |
| `scripts/archive/benchmark_latency.py` | 133 | `5d59a217ba4ef1305fa874764c6baa0424807dcf6d71c67eb213a7a5bf5f6932` | R: Historical MT/TTS means labelled pipeline target; archived import path broken. |
| `scripts/archive/benchmark_omnivoice.py` | 93 | `f87482fad5074ca9a0f1233df0fd1066f6639e6cd23aec0daf9b6ea0a3bc60df` | R: CPU-only old clone probe, moved ROOT breaks default-reference read at parser construction. |
| `scripts/archive/benchmark_ssbd_mt.py` | 101 | `4fdd2a3a8a73a691ed6ee1ccb3067e90f1ca9e1cc66c83024c560ed035b7befe` | R: Growing-prefix A/B latency, no separate quality grade; retain negative-result research. |
| `scripts/archive/eval_bilingual_voice_control.py` | 120 | `c36294493cc1775974e414750cf29d768c65f3df5c7c3cacaa1e4ec77ba96cb4` | R: Archived ROOT stale; reference string reused as row then row[id] TypeError saving examples; summed medians. |
| `scripts/archive/eval_sk_direction.py` | 173 | `ed41594ff0f97ebd49354ceb9142732dd5a90b7bc14f092634a8edd2ddac046d` | R: Repo ROOT repaired here; MT reference falls back same model, fixed overwritten outputs; proxychrF. |
| `scripts/archive/eval_stt.py` | 36 | `49a5ada36afd5da9880a3997ec986bcdb4a039aa81f986a6059427f091a3c754` | R: Archived ROOT scripts/, stale conversation import; fixed18clips, whole clip ASR only. |
| `scripts/archive/eval_stt_parakeet.py` | 31 | `b90f13d603d5b5b12417c49ad6d4d4a8640b7c41795863748e77531f031d81d0` | R: Archived ROOT defect, optional onnx-asr; warm whole-clip ASR only. |
| `scripts/archive/eval_stt_tune.py` | 27 | `68533eb6b71e9676f958312e4fd9f73bcbc24a6a070a4b7f556ef50038828b04` | R: Archived ROOT defect; raw argument/clip assumptions, stdout-only evidence. |
| `scripts/archive/generate_charts.py` | 349 | `b6bcc98f15edf028751b5710adc4015ba88d38abd72d210cb0af457c4fb42f75` | R: Hardcoded baselines, fallback measurements and zero-artifact subjective scores; quarantine historical figures. |
| `scripts/archive/live_direction_probe.py` | 181 | `0c06c2aafb4336aa73ff34ede88c730401bb5f58b94bce1fa2634744ae1dfac5` | R: Unauthenticated wss, first ack not model readiness, ignores errors; archive move breaks callers. |
| `scripts/archive/male_alldata_denoised_benchmark.py` | 128 | `1aa1cbbddbdb02552056e9fd269d9188cb46dd4a5fd85365c0e89d3d8dec3bfe` | R: Synthetic comparison, actual small model vs small-sk claim, inference run twice, stale ROOT. |
| `scripts/archive/parse_test_results.py` | 203 | `c21798a0e2abb23d7c2d7da08cb2862b844954702e1428e96122113838717a78` | R: Old F5 report parser, silent partial fields/literal legacy labels; not current receipt schema. |
| `scripts/archive/seamless_matrix.py` | 82 | `cc1b9765b16a01e3010064cca6db22ec303e0e426a3f5d4bbf7911ecb0883a0f` | R: 30s CPU chunks vs Opus-derived refs, stale ROOT; noncommercial experimental reference. |
| `scripts/archive/segment_generic.py` | 90 | `cc00040167f52fbfcd5253aff60d8d6bae0c58a6179e8bce894bccaa78da7eb4` | R: Forces audio cuts to sentence count with longest split; alignment must be manually reviewed. |
| `scripts/archive/segment_sk_dataset.py` | 111 | `3be9d16c1f3a03d9b2e7951c35594da95d154f876bff30436cc386186edfed11` | R: Forced-count heuristic, hardcoded temp source, output manifest omits wav/ subfolder. |
| `scripts/archive/streaming_audit.py` | 94 | `d05f244259e33be9e912c9c1e62d4413aebd9c36ee3b1d60cbc0582f2f0b5aea` | R: vad_to_text is first transcript since session, not parsed VAD-close delta; stale ROOT/probe. |
| `scripts/archive/stt_model_benchmark.py` | 90 | `4322c8cd20d5acc0f1323af8b54c0af7020a8add843eff319598304770357429` | R: Synthetic passage, order/cold-state confound, personal comparison not persisted; stale ROOT. |
| `scripts/archive/stt_parakeet_spike.py` | 82 | `d7fa6d25331a5dec79e9ed7d52f74aae0823a85d4e97988307f1430be982e777` | R: Language-keyed metadata selects last clip, partial windows can compare full reference; cold run. |
| `scripts/archive/stt_speed_test.py` | 72 | `7bf6e8d4a5cf6a199a91a7b638ef03999ae384ba6c71a574ebf59b996e20829d` | R: Synthetic speed-only, stale ROOT; absent input returns success. |
| `scripts/archive/test_backend_pipeline.py` | 256 | `af4b608602617cda5c5feedde0461026eb8a67a095285885c593f083ff0eb9c3` | R: Piper second positional arg misused as output path/language; truthy tuple as success, no saved output. |
| `scripts/archive/test_full_pipeline.py` | 261 | `a1e46e649986242072ad033dbd7d52b7ff4f81d34f4480faf5be1f5e6c921ba6` | R: Expected text substitutes emptySTT, Czech TTS proxy, model loads counted; archive. |
| `scripts/archive/test_pipeline_optimized.py` | 130 | `2eee8ee24a3bad015b5d73f6b0d01e4f978446c34d258ba6f573de87970ef1ca` | R: Stage sums and unsupported streaming latency extrapolation; archive. |
| `scripts/archive/transcribe.py` | 224 | `a1281bd5d81d3aa631bb8a1f83c13592151d46312d4423ff113eb017439aacf0` | R: Owner-hardcoded video/output, old Whisper/MPS pipeline; SRT concatenation does not offset chunk timestamps. |
| `scripts/assemble_train_corpus.py` | 62 | `52db81c43d7a7d4134f020fd8e38b781a836ae9296a0d06665382a60e0b26d92` | R: Synthetic corpus assembler, requires local manifest/WAVs; fixed temp output overwrites. |
| `scripts/audio/setup_linux.py` | 309 | `f33da3715c9ea8a56d6d2dc4ab74c9585d8a30dae41ccbb41831a4edafdb04d4` | R: PipeWire module IDs missing, no-op removal/failed rollback; hardware route untested. |
| `scripts/audio/setup_macos.py` | 232 | `e686d066d420f86cdcf1f572af6b3c94a23ff5b5aaaee9990936a2568412e43e` | R: BlackHole install/detection, hierarchy parser/1-2channel assumptions; real call route untested. |
| `scripts/audio/setup_windows.ps1` | 218 | `93c2158d622b4666da5ac711cdade67834828774b364ebbf4f452e5b2b003958` | R: Temp directory missing, PowerShell5.1 incompatibility; installer/routing untested. |
| `scripts/bench_nemotron.py` | 89 | `f4edf77a84c10fb4d84ef4ff0d8d93213c0d00c52d5531cca81357a1eb630b2d` | R: External C runtime, hardcoded temp defaults, duration from WAV bytes includes header, missing revision receipt. |
| `scripts/bench_new_models.py` | 163 | `5e35e1e994e8be6b36eaee75e4ff66505a49e83407b9ac11127a8cc1f582423a` | R: Only Seamless implemented;4 explicit TODO adapters; chrF vs Opus-derived reference is proxy. |
| `scripts/bench_stt_beam.py` | 138 | `247596bdd0a55e2c1b8836b9a28cf1fa8fcd7664d3bead030f711bd6a6d09f63` | R: Beam/chunk/prompt ASR sweep; uniform chunks differ VAD, fixed outputs overwrite; no full route. |
| `scripts/bench_stt_synth.py` | 107 | `57ea08fc60d571e17b88746e7db5665afe7b364752ed54494f548ee4e5668f87` | R: Useful resampling/VAD ablation; Czech/SK synthetic voices, cannot establish real mic quality. |
| `scripts/bp.py` | 637 | `0d49b467bc106de627fce1c5be58fc9b23e2d1d01cfe2196563503ae7481554b` | R: CLI dispatch, selftest and engine quality-gate defects above; model-harness parser reproduced. |
| `scripts/build_demo_charts.py` | 317 | `132f6a04bbaaca8ec7b9749e64e1740809f702e7d610071ebab0a23041c420fb` | R: Derived figures, assumed streaming Gantt and capped-row STT denominator defects. |
| `scripts/build_recording_set.py` | 89 | `26bac0b0afb19f8c7baca4fd86b3f281df4ce4bda210c54f1d6a7ec319e967d9` | R: 15 conference+60 filtered NTREX phrases; mutable main fetch, no source digest; generated script overwritten. |
| `scripts/bulk_hq.py` | 144 | `dc7d4dbd4edda3db66a8ae2e4ce1341dcbe2966f3bcf49ce393a77cb06f90ec8` | R: MPS-only synthetic corpus; English mode still incorporates Slovak NEW passages; destructive manifest pruning by start. |
| `scripts/conversation_sim.py` | 195 | `4dbbb60a88c9d2e403f363f48aa1e3c7f79bf9722265ada34b10fc05f14a1bcd` | R: Offline sentence simulation, forced count alignment, legacy recordings; MT batch average not first-word latency. |
| `scripts/convert_models.py` | 53 | `8354ca35baa88a42b9fd1b3aedd3d5aba7811b9b527da18ec3e45bbef824b16e` | R: Eleven hardcoded MT routes; model.bin-only skip; fixed converter venv finally deletion; unpinned. |
| `scripts/convert_whisper.py` | 23 | `124486414ac18d4d2c872eaea0164fa257a06b9661ae67b4220899652edfc789` | R: Conversion saves tokenizer/features, force overwrites; unpinned source and no argument validation. |
| `scripts/demo_audio.py` | 124 | `8eafaf837c19b7a686d4a2e700cc4777936011e8be14c81b17545dee32cae9bd` | R: Offline EN/SK smoke, beam1 and personal voices; wall includes loads, not live onset. |
| `scripts/demo_conversation.py` | 287 | `55ba969e86373a044e7a704167c6755c6631c89e3aef8828ab13972dbf57f75f` | R: Offline/synthetic comparison; generated HTML hardcodes CPU/noGPU/about1s claims independent of actual device. |
| `scripts/demo_preflight.py` | 154 | `8fa9190e3ad11445b343b79fbe2b84b4bac3c0552ec03e0d6cb7986615061bfe` | R: Asset existence only; owner-local heuristic, no asset integrity/quality/actual route test. |
| `scripts/e2e_ensk_new_voice.py` | 65 | `b97e5dcf8e4f655f2658f56642b44aa346ea71c26306ff7f53012166716eca05` | R: Hybrid-only offline legacy test; TTS truncates600chars; optional dependencies, partial output. |
| `scripts/engine_ab.py` | 158 | `af7f09f8c85640aac2927e640493e3cef0598aa6af937af72b7c03c83272d47e` | R: Synthetic TTS/STT diagnostic; hardcoded MPS/venv; cache not fingerprint-bound, WER recognizer-dependent. |
| `scripts/fetch_whisper.py` | 15 | `21de59a4709380e1c14743bd0771ffb90e241fa60d3fae7b67155faa16477e8d` | R: Unpinned snapshot downloader, setup-only mutation; add revision/hash provenance. |
| `scripts/finetune_personal_voice.py` | 285 | `087ae8bcb91d94250d1696ccef7490be3470090ffc8031e71db0fa949db9aaf6` | R: Offline CPU trainer; fixed version0 callback vs newest-version export logic; WAV-only inputs; export provenance. |
| `scripts/gen_cert.py` | 33 | `3cf4313b20107b01e564410c05a0aee6a586161a8f71885c8c1ceca160bba47f` | R: Certificate generation at import overwrites fixed cert/key; key permissions attempted; explicit CLI/idempotency needed. |
| `scripts/gpu_bench/README.md` | 17 | `8bea70bb190074f8787dd81fdc924c56c830e851c293c2681e47ad9542360fe0` | R: historical TTS/Colab procedure, ear-grade limit explicit; consolidate supported entrypoints. |
| `scripts/gpu_bench/engine_smoke.py` | 34 | `f4e2bc92ad388b98519bfee0eca80b49942011d31d61b40164de12b0e120bcf3` | R: Shipped TTS class only, first clause chunk distinct from route latency; no hashes/revision/quality. |
| `scripts/gpu_bench/gpu_bench.ipynb` | 39 | `f776ad4e777e074d0fb2ace3693bb3048bbcc859f41fe722cfb0c79e5a33d0ba` | R:4cells upload/extract/setup/run/download; consumes legacy stub installer; historical. |
| `scripts/gpu_bench/gpu_bench.py` | 172 | `f3e99d49dece100a6be2515d5d87e0ee4e0b3003b222334326595baec197806e` | R: RawTTS; setup installs inert stubs on failure, unpinned deps/models; marginalRTF is not onset latency. |
| `scripts/gpu_bench/kaggle/kaggle_bench.py` | 51 | `8bc02d33c61ef8846106f51fa204492131ce6b6b27fd33bf4aaa71ffaad76927` | R: rawlegacy stages/check=False failures/mutable weights; archive. |
| `scripts/gpu_bench/kaggle/kernel-metadata.json` | 3 | `aaeeb21fed48598d5898c253bb7aceb0e77c976e8fa3ae375f4746be5f5b3a27` | R: privateGPU/internet-enabled job config, owner-specific datasets/kernelIDs; not public setup. |
| `scripts/gpu_bench/kaggle_canary/README.md` | 42 | `9e08518c30cdb6bfbca7a28c5e59898c3c532afa3dfd35b9c90e90148fca8728` | R: bounded remote-only instructions, acceptance limits explicit; local lightweight tests ran, remote not run. |
| `scripts/gpu_bench/kaggle_canary/kernel-metadata.json` | 13 | `950d7adc298f7cef61514a74d178dc38e8dd9af7193891eaa0e097a88c93cc0c` | R: privateGPU/internet-enabled job config, owner-specific datasets/kernelIDs; not public setup. |
| `scripts/gpu_bench/kaggle_canary/rescore.py` | 27 | `f8e9959b5d9a7d9e47d59c48ccd5ca2a6f733522d534e11899c41a3969c6b7c7` | R: saved-string rescore/sourceSHA, no inference, historical floor schema. |
| `scripts/gpu_bench/kaggle_canary/test_validate.py` | 42 | `203a136f1a5168217b40ca7cd4cc3bae7db59f58ebcb8dd03b6b2038043edf69` | R:6scoring/acceptance tests pass; noGPU/model check. |
| `scripts/gpu_bench/kaggle_canary/validate.py` | 126 | `171d4d22ef09272c7d642724591212aca19122459888b9216feb88b952f22ed1` | R: proper WER/reference denominator and exact task coverage; pinned resolved model revision, ASTunscored. |
| `scripts/gpu_bench/kaggle_conc/job.py` | 229 | `a1e480ef9be0a7a81202a68d0e032bf971339dc61c987d9af11b3c45d0461660` | R:1/2/4/8clients common account/inferred lag; skipped-cap complete and client failures require scrutiny. |
| `scripts/gpu_bench/kaggle_conc/kernel-metadata.json` | 15 | `d517f1aaf7785ff5f520479eb493a43a4ba6de9500b81d998c7adc4496825031` | R: privateGPU/internet-enabled job config, owner-specific datasets/kernelIDs; not public setup. |
| `scripts/gpu_bench/kaggle_data/convert_opus_mt_to_ct2.py` | 61 | `91771fd5ca1e3efb4672fb52859da7cd8234868a963007f56487f1b58281e56a` | R: converter saves tokenizer; force overwrite/unpinned source; duplicated runtime helper. |
| `scripts/gpu_bench/kaggle_data/cs.txt` | 1 | `bab6e21bd2da3612195872baeeb7d9ec61a7958d947a776d7672e52b61ed62e9` | R: short clone-prompt transcript, not a full ASR/MT quality reference. |
| `scripts/gpu_bench/kaggle_data/dataset-metadata.json` | 1 | `bd1d902f5b6d86b0b3f7490b09cc2d04a87f9e04c31fde011c7edb25b7b21550` | R: owner-specific datasetCC0 declaration; verify consent/privacy independently before publishing. |
| `scripts/gpu_bench/kaggle_data/en.txt` | 1 | `d3af428d8732bacd3ee35f3540138e01f517689ed5760d6a46e703ea30ec52b4` | R: short clone-prompt transcript, not a full ASR/MT quality reference. |
| `scripts/gpu_bench/kaggle_data/gpu_bench.py` | 172 | `f3e99d49dece100a6be2515d5d87e0ee4e0b3003b222334326595baec197806e` | E: exact SHA match to reviewed gpu_bench.py, including dependency stubs. |
| `scripts/gpu_bench/kaggle_data/load_bench.py` | 302 | `b312f40654246ffd350d96e6b329aafa2ffd1a2dbacffd3a96b0888499e5f62d` | D: full current Cascade plus diff; older rawOmni-only API, no steps/batch/engine support. |
| `scripts/gpu_bench/kaggle_data/sk.txt` | 1 | `92d685cb66a1fa0fdd38d66a2caff3adcce77cfb5eba61ad275ed8810226c3a7` | R: short clone-prompt transcript, not a full ASR/MT quality reference. |
| `scripts/gpu_bench/kaggle_data/tts_tune.py` | 143 | `bf9a49e955ee68cdde33c553a7357370ed5b843a3c26e3fba892aa0fbd691d5b` | E: exact SHA match to reviewed tts_tune.py; neighboring older Cascade remains compatible. |
| `scripts/gpu_bench/kaggle_de-en/job.py` | 398 | `b8d4456468c9a0ab4528c129f31bfe9b7e1d427e5cfeeb3bb2579b1d85491482` | D: complete full_route baseline+diff; invalid English placeholder for source; inherited CSmetadata/offline/provenance issues. |
| `scripts/gpu_bench/kaggle_de-en/kernel-metadata.json` | 17 | `950f62fd23dba47225f689a642bfa389b9583a48e377c598d14fa6c12f490b5a` | R: privateGPU/internet-enabled job config, owner-specific datasets/kernelIDs; not public setup. |
| `scripts/gpu_bench/kaggle_en-de/job.py` | 398 | `71ca91089eac0d0c29a6e17bb798a2c5da00e4c9d94a1d22dd4926794c86f0cd` | D: complete full_route baseline+diff; English input valid; inherited CSmetadata/offline/provenance issues. |
| `scripts/gpu_bench/kaggle_en-de/kernel-metadata.json` | 17 | `fa567405ffacc589d528445f99efae04e0bf19eda2025d50b1f26a7f24054852` | R: privateGPU/internet-enabled job config, owner-specific datasets/kernelIDs; not public setup. |
| `scripts/gpu_bench/kaggle_en-fr/job.py` | 398 | `a987946604b81988409df56fb2f488f994f731eebbda3eaace69d493cf93203b` | D: complete full_route baseline+diff; English input valid; inherited CSmetadata/offline/provenance issues. |
| `scripts/gpu_bench/kaggle_en-fr/kernel-metadata.json` | 17 | `65591fce56f543a078bb8874c66d1a489b61d1a3c8ae0d1771b25eb7a81fd147` | R: privateGPU/internet-enabled job config, owner-specific datasets/kernelIDs; not public setup. |
| `scripts/gpu_bench/kaggle_final/kaggle_final.py` | 34 | `691fd4c6c6ba97416c2ecbba628ee33034430d43743233f3a3d5ec8bbf3911cf` | R: clones mutable publicHEAD, ignores smoke/ramp errors, no exact reproduction. |
| `scripts/gpu_bench/kaggle_final/kernel-metadata.json` | 1 | `b64bcada7a401bc81f3d9509ff6419666eb45f516239f6ed56b306cf7811ee59` | R: privateGPU/internet-enabled job config, owner-specific datasets/kernelIDs; not public setup. |
| `scripts/gpu_bench/kaggle_fr-en/job.py` | 398 | `d943b63f0160fa9ea21831c4941259c011684c6367f4aaf7feeac4b98ecd81ca` | D: complete full_route baseline+diff; invalid English placeholder for source; inherited CSmetadata/offline/provenance issues. |
| `scripts/gpu_bench/kaggle_fr-en/kernel-metadata.json` | 17 | `c89707898c759181dcbfefea4d764c1bf951e54a6eddbae206ebbcdb67d102ff` | R: privateGPU/internet-enabled job config, owner-specific datasets/kernelIDs; not public setup. |
| `scripts/gpu_bench/kaggle_full_route/build_package.py` | 63 | `53e3cb131aabb5e240c401d1f625567d2b350c414e1b74aa9cfcea89831b622e` | R: fixed temp dir rebuilt, missing allowlist paths only warned, no model/provenance package. |
| `scripts/gpu_bench/kaggle_full_route/job.py` | 398 | `041e9cfd2081a5b6c43a799bc835c0bbf52630f312b658450d181e43a6771ef6` | R: newest productionWS warmup+3repeats/onset timestamps; fixture/offline/tokenizer/provenance defects. |
| `scripts/gpu_bench/kaggle_full_route/kaggle_full_route.py` | 190 | `8b8ea4140b7112af6945b108278a526631acccfafbcedc833c9b67ff51883491` | R: direct pipeline, absolute emission timestamp quantiles mislabeled latency. |
| `scripts/gpu_bench/kaggle_full_route/kernel-metadata.json` | 17 | `096f0f6df12abc99e7e7efa369d85b037494cf636c800a9887e2da5591d262ac` | R: privateGPU/internet-enabled job config, owner-specific datasets/kernelIDs; not public setup. |
| `scripts/gpu_bench/kaggle_incr/build_package.py` | 52 | `0b042318b129764da29dc2d8741911f1db63e93b4dd2cb283168131c590aa367` | R: allowlisted code/private90sclip/generic voice; fixed output overwrite, stale route docstring. |
| `scripts/gpu_bench/kaggle_incr/job.py` | 239 | `43dd09d9ff7e4bb6479abcec46703f421fe269ba78dc72323e37fac6f1d04e3d` | R: productionWS A/B/C/D, patched timing logs; complete accepts skipped-cap, meaning ungraded. |
| `scripts/gpu_bench/kaggle_incr/kernel-metadata.json` | 16 | `8e70a7704bb3cc34bb9a71ae7f35576fd021e8bbfabce31b1c9ef03998b39da4` | R: privateGPU/internet-enabled job config, owner-specific datasets/kernelIDs; not public setup. |
| `scripts/gpu_bench/kaggle_lang/build_package.py` | 53 | `934366815abfc91d51ee2026ab7b19b3a101a94db4e71352248c10c9f7879250` | R: three60s clips despite90s docstring, generic voices and allowlist, no source digest manifest. |
| `scripts/gpu_bench/kaggle_lang/job.py` | 228 | `10417f2f77ed83d0561f1d8856df3461df55fb25fb0db49268bfb479b6a905bf` | R: correct EN/SK/CS fixture routes, turbo override; inferred listener metrics, skipped-cap acceptance. |
| `scripts/gpu_bench/kaggle_lang/kernel-metadata.json` | 15 | `ff1c0edabc1e902fdbf54e6390f062e60298d999a3770d2a8736bf5d3b796879` | R: privateGPU/internet-enabled job config, owner-specific datasets/kernelIDs; not public setup. |
| `scripts/gpu_bench/kaggle_load/kaggle_load.py` | 27 | `54c547ed2cf19cdf16aa072ca60bccec2327ed30422fcf2ed3a8de2c7ee84700` | R: remote rawCascade/converter/unpinned versions, distinct from productionWS. |
| `scripts/gpu_bench/kaggle_load/kernel-metadata.json` | 1 | `3609312f13dc9eb52a7bd8b1ac73bb5b5e7cf124a82d898b1cc5445293ef39d3` | R: privateGPU/internet-enabled job config, owner-specific datasets/kernelIDs; not public setup. |
| `scripts/gpu_bench/kaggle_omnivoice/job.py` | 168 | `bd19b33e32dc45575567368dbf6e4a6af7090115ea872bcc451a805b3dd65394` | R:1case/langTTS, CPU fallback, status never completes, per-case errors exit0, noASR/listening. |
| `scripts/gpu_bench/kaggle_omnivoice/kernel-metadata.json` | 17 | `f26bdd22af16fb5d8a03e370cf7eb2c937671c2a5b3d43e4d9b6af472faadcb9` | R: privateGPU/internet-enabled job config, owner-specific datasets/kernelIDs; not public setup. |
| `scripts/gpu_bench/kaggle_route/build_package.py` | 50 | `1ee64227d513cf166301e2bbd449f3c227a96882bb459326c1f081223f71c40c` | R: allowlisted code/private90s clip; second denylist check tautological, no code manifest. |
| `scripts/gpu_bench/kaggle_route/job.py` | 302 | `29d62c8cc2fef67cedab526d33bd95a257c22d77cd4448152ed6540831490b12` | R: realWS cloned route; patched licence/revision, relaxed deps, remote assets, no listening grade. |
| `scripts/gpu_bench/kaggle_route/kernel-metadata.json` | 16 | `9f0a04dd3863e783c4cc9ed28c71d44ed02d7d3255f3750a855e8f58cac8d697` | R: privateGPU/internet-enabled job config, owner-specific datasets/kernelIDs; not public setup. |
| `scripts/gpu_bench/kaggle_session/README.md` | 60 | `ed76ad8ee6ed214150ad7e67cf27c04634bbdb7485b22a7b019567477992b550` | R: bounded remote-only instructions, acceptance limits explicit; local lightweight tests ran, remote not run. |
| `scripts/gpu_bench/kaggle_session/job.py` | 229 | `bc1cbd7c6ced615f49051cdbdbc1fbc8e5d729c244e1b5a49a24779bdffbbdf6` | R: pinnedGPU TTS12rows+synthetic queue, correct disclaimers; normalization overwrites original sample-rate metadata. |
| `scripts/gpu_bench/kaggle_session/kernel-metadata.json` | 15 | `140a4c8e52bfa0d76bdb7a0f51299236273961af679fe9913c00a11de168c775` | R: privateGPU/internet-enabled job config, owner-specific datasets/kernelIDs; not public setup. |
| `scripts/gpu_bench/kaggle_session/test_replay.py` | 57 | `1df2201eb20a1cc30e60e75613fb3cf2c1382f1e2aa79b700fc643120b00c3e7` | R:6lightweight queue/normalization tests pass; missing original-rate/hash assertion. |
| `scripts/gpu_bench/kaggle_tune/kaggle_tune.py` | 27 | `7de4c28b062cefa1c4edd522d2bbed27a950c1b76b9c268511f048887a8cd7b0` | R: remote older copiedCascade tuner; unpinned deps, historical diagnostic. |
| `scripts/gpu_bench/kaggle_tune/kernel-metadata.json` | 1 | `0dcafd9391bcd91fc5a8ee21db5140384d89ae732d45bcc3e923165f51899806` | R: privateGPU/internet-enabled job config, owner-specific datasets/kernelIDs; not public setup. |
| `scripts/gpu_bench/load_bench.py` | 350 | `40e549b84e17b2c68d4b749d0926fdf94cf4c984dca1a941b42ba55ea5726ec1` | R: Raw shared cascade + virtual timeline; WER vs idle only, unhandled worker exceptions and indefinite batch wait. |
| `scripts/gpu_bench/modal_bench.py` | 78 | `24590c0b3d29b037a0fe2ce984239af4d1e97aa40e48c76a6c760f8b90c78843` | R: Remote private-ref upload and cost path, mutable model/source dependencies; not run. |
| `scripts/gpu_bench/tts_tune.py` | 143 | `bf9a49e955ee68cdde33c553a7357370ed5b843a3c26e3fba892aa0fbd691d5b` | R: Self-ASR synthesis quality proxy; incompatible namespace with current sibling Cascade; dataset copy differs context. |
| `scripts/grade_library.py` | 213 | `d71b6f58441e46b86dcd702b9fd361f7361d5132e4d91a8a487a8d4366bd7d69` | R: Advisory/ear grades separated; illustrative similarity threshold promotes pass; 8-bitPCM signed decoding wrong. |
| `scripts/i18n_tool.py` | 224 | `26ac2179be0ebdb1661cccb6aceedacf954b1d2c8cf9821feb3c4d8fcd9cb3ed` | R: Placeholder/provenance checks; remote translate sends UI strings; no network/mutation run. |
| `scripts/language_packs.py` | 472 | `e2de9565a6cdf0044aa522bf12bdf3f157a278f0a0ef941c57f43d0bc03a046b` | R: Improved fail-closed review fingerprint; production-setting bypass and licence inference defects. |
| `scripts/live/backend_vad_stt_test.py` | 138 | `9e45b071543557287c7c5fcf1fc64287d3b3c50778af632d345c86d2faa2483f` | R: Unpaced offline VAD uses wall-clock silence and per-frame peak normalization; not production timing. |
| `scripts/live/batch_stream_test.py` | 140 | `9ea9398d75612a2b73e2226e3d6851c8eb5fd8b44a75de6dca9fb10113e5aa00` | R: Default half-speed sender, Stop before drain, no concurrent receiver during send; not1x live gate. |
| `scripts/live/fragment_quality.py` | 21 | `582218739fbea05192b1950e24ae743e214efdf521186a2a581dccc6ce5885bb` | R: Correctly labels chrF++ vs same MT unfragmented output; fragmentation proxy only. |
| `scripts/live/interrupt_smoke_test.py` | 146 | `f8eed05351103f08a2126df64dcb19405a219ff124f13da76bb18d1d8ce733f3` | R: Auth WS output and2s Stop-window check; explicitly not model cancellation proof; fixture/account/cert dependencies. |
| `scripts/live/paced_replay.py` | 185 | `ee59cc73292467db8ffed6dd8c3d08f90fb443b55a387b6ed756191de0bbb53c` | R: Real paced WS route but inferred listener model, fail-open error/empty output and unawaited receiver. |
| `scripts/live/piper_pipeline_test.py` | 390 | `d00d6e55e95857c631eb056c8cb3e8d980d161df2e4c22b85d75bdd051714630` | R: Offline VAD copy uses wall-clock time; hardcoded MT correction, assumed output rate; not production acceptance. |
| `scripts/live/summarize_replay.py` | 22 | `83031354de6867d90964bd99abbad82173635f0d497ee458a34c9b8c069f70be` | R: Listener-delay proxy; empty output fails; incremental-span precision not independently checked. |
| `scripts/machine_listen_qc.py` | 153 | `2f92b7ad20ae5820a0dfee01938cad1f9140b89c7ed73257cf06a669d4e4ae25` | R: Audio thirds not word-aligned, recognizer/acoustic proxies need calibration; legacy refs, no human meaning proof. |
| `scripts/make_reading_script.py` | 75 | `2cd8aa488b72cdba3a8427303787e47f7a82da61c3710bfcb5f74338c8943891` | R: Overwrites proofread SK column with machine translation; preserve corrected scripts/version outputs. |
| `scripts/meeting_live.py` | 190 | `187b02326770ae18cc657c617f64b9df0c062de6cee9bc267b9897c8d0ae5344` | R: Synthesized input via missing archived probe path; reconstructed sequential meeting; collides simulation output. |
| `scripts/meeting_sim.py` | 197 | `9576a91cf7bb3ada2d86d059b094f66c01c9fc669a7351588dffc3e949730d4c` | R: Assumed STT tail substitutes measured STT; simulated overlapping playback, no sequential player bound. |
| `scripts/model_harness/README.md` | 14 | `93b377efb1673e24af017ecda8c36b166299360252bec1a8b8acc8d18d35397c` | R: examples/adapters/output never-overwrite promises contradict code. |
| `scripts/model_harness/run.py` | 165 | `34d038efa385ec3a1341cf20d395a2a74aff6cfefe3f2c4a00b94be0c3ecd104` | R: unsupported adapters, truncatedrefs,declared-onlyVRAM,overwriting paths; parser failed before inference. |
| `scripts/pipeline_latency_probe.py` | 97 | `270cd3b5dca8aad4ab4bb652fa84fff335adb2248b7acc8fcbba4d81a3c22fa8` | R: Missing archived Parakeet script/legacy recordings; generated30s probe file unused by child. |
| `scripts/praat_voice_report.praat` | 19 | `4970caabafc493cf356b517b293f6fa0fa5b42b7d174d9dd9d6541f6b9383b52` | R: acousticF0/jitter/shimmer/HNR extraction only, no calibrated meaning/identity gate. |
| `scripts/prepare_voice_corpus.py` | 240 | `cfce62741355e1f11515eae9fbf06cb2f517bbd9a7ac59ccf278c4acd32f8f22` | R: Dry-run temp mutation, normalized-stem collisions, heuristic near-duplicate risk; documented --merge absent. |
| `scripts/record_reading.py` | 79 | `87978fadc0dc16bd10a1712455509cd7b2da1a185cdef4ecc4669b2ab7e0dd1f` | R: Interactive recorder with resume/redo, RMS/clipping warnings; human reference/audio acceptance needed. |
| `scripts/setup.py` | 99 | `b9e09ad4ab034d0452d0e63342194f4d276072b596dfab40942b3adf75383db8` | R: Cross-platform baseline installer; hardcoded model bundle/grade store; no fresh install executed. |
| `scripts/train_status.sh` | 68 | `5a6088523e6cef0292166a8481884e83a310352e2f4c3cda2734cbdc45830c4d` | R: hardcoded owner/version0 paths, timeout200s torch checkpoint load; heavyweight status command not run. |
| `scripts/ui_manifest.py` | 82 | `acc355e0ef5f336c3716147307bdc2446f1c964b2390a8360bf9317325130b74` | R: Deterministic generator/check passed; declared routes are inventory, not verified support. |
| `scripts/update_voice_lab_library.py` | 422 | `6c371bfadada7e8ed8ae6ca82357e3d113e417a191a263d2be96b760d79c4817` | R: Local media and grades index, historical renames; may expose owner transcripts if published; local assets required. |
| `scripts/vad_close_probe.py` | 103 | `a5db0e396a2075d8b787958a17100d499655fcde3751a64f633888a3444c8a97` | R: Obsolete auth/config/result/TLS contract and3x audio pacing; repair/archive. |
| `scripts/voice_similarity_qc.py` | 192 | `d71d3df354a53f9995e4f143cf525032d1243eb545fcf40fff79d92837762b77` | R: Illustrative resemblyzer proxy separated from ears; legacy refs/optional Hybrid, no automatic identity proof. |
| `specs/001-realtime-cross-platform-translation/plan.md` | 78 | `a73ae8f1f98ac80df172b17af37a1ec46b8792c46e6404294f218b1146393643` | R: historical superseded scope, file-level archival banner absent; do not execute legacy tasks. |
| `specs/001-realtime-cross-platform-translation/spec.md` | 140 | `2b846543d70c6d3c16c1c7250fc80770225daecf36e70d9433122149a034dcd9` | R: historical superseded scope, file-level archival banner absent; do not execute legacy tasks. |
| `specs/001-realtime-cross-platform-translation/tasks.md` | 289 | `817544853ebf05902f95abe3b70afe0063ec5702f703984c5388abb31109a94e` | R: historical superseded scope, file-level archival banner absent; do not execute legacy tasks. |
| `specs/002-pwa-pip-demo/plan.md` | 31 | `bfb4526ac1c727968f96c9be568604ed24018219fb5003b09fb1448f49785e69` | R: historical superseded scope, file-level archival banner absent; do not execute legacy tasks. |
| `specs/002-pwa-pip-demo/spec.md` | 66 | `d5242ab96bb5f6b5f6e050bf3bfb71fb51fcce5e2155546b8a868a207f90e201` | R: historical superseded scope, file-level archival banner absent; do not execute legacy tasks. |
| `specs/002-pwa-pip-demo/tasks.md` | 27 | `2128c0a1b28d98d36870c2f8965de87e7b5ee8668f99fb83465e43787ce59e38` | R: historical superseded scope, file-level archival banner absent; do not execute legacy tasks. |
| `specs/003-end-to-end-completion/plan.md` | 116 | `500067f5420c77aec3288423543094be13699c958c5ad9614c50dc739f6141e5` | R: active-labelled draft needs requirement/status/path/latency reconciliation. |
| `specs/003-end-to-end-completion/spec.md` | 286 | `88099002cf26e51a0e5259d60a725899b942d89097aca886c7ee4bce45c2c1e0` | R: active-labelled draft needs requirement/status/path/latency reconciliation. |
| `specs/003-end-to-end-completion/tasks.md` | 152 | `0c77c929dc4af6133ee6d576584ae679370dd40fca7c1279dbd5f029ff318669` | R: active-labelled draft needs requirement/status/path/latency reconciliation. |
| `specs/_archive/README.md` | 29 | `8b182d4e189c06dcf74588faae4e65dd196c0c1f20e626b8e02f4e6b535a7867` | R:001/002archive declaration,003active; original files remain outside archive without banners. |

Coverage: 189 files, 26,180 lines including equivalent copies.
