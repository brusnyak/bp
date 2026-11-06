# Kaggle authentication and evidence repair — 2026-11-03

## Proven in this session

- CLI 2.2.4 already installed. `KAGGLE_API_TOKEN` is nonempty in `~/.env`, but
  neither uppercase nor lowercase token name was in the inherited process env.
  `with-env KAGGLE_API_TOKEN -- kaggle kernels list --user yegorby` succeeded.
- Created `~/.kaggle/access_token` exclusively, mode 0600. Plain `kaggle kernels
  list --user yegorby --page-size 3` then succeeded without a wrapper. No legacy
  kaggle.json needed. Future OpenCode shells can use the CLI directly; do not
  weaken OpenCode's permission rules. Restart a client if it retains a stale error.
- Existing private output `yegorby/bp-tts-gpu-compare/6` was downloaded and hashed.
  Both saved SK fine-tune outputs score 3/11 = 0.2727 with reference-normalized
  WER. Historical 0.2143 divided by hypothesis word count (14). This is a saved
  string re-score, not renewed inference. Reference/audio completeness unknown.
- Submitted a new private kernel, `yegorby/bp-canary-sk-cs-validation/1`, bounded
  to 1200 seconds. Only existing dataset `yegorby/bp-gpu-bench-refs` attached;
  no new voice recordings uploaded. Used the model card's Transformers path,
  avoiding the old missing-NeMo-Lightning path. Five output rows, no errors,
  Tesla T4, 133.89s total including remote install/model load.

| Task | Short clip | WER | Inference |
|---|---:|---:|---:|
| SK ASR | 7s | 0.3636 | 4.6052s, first/cold call |
| CS ASR | 5.6s | 0.2857 | 0.2255s |
| EN ASR control | 7.25s | 0.0588 | 0.2583s |
| SK→EN AST | 7s | unscored | 0.2036s |
| CS→EN AST | 5.6s | unscored | 0.1524s |

Translation has no reviewed English reference; do not apply source ASR WER to
English output. Czech output mistranslates the square name. No route promoted.
The common Slovak trailing phrase in multiple outputs also makes reference
completeness a necessary next check, before interpreting it as hallucination.

Peak PyTorch allocated/reserved: 1.828/1.842 GiB; not total process VRAM.
Checkpoint revision and dependency versions are recorded in
[evidence/kaggle-canary-2026-11-03.json](evidence/kaggle-canary-2026-11-03.json).
One inference per task, no warm repetitions or first-audio/full-route benchmark.
V1's executed source hash is recorded; subsequent local acceptance hardening
requires all five expected task rows and fails on missing clips/references.

## Reproduce without secrets in prompts/logs

```sh
kaggle kernels status yegorby/bp-canary-sk-cs-validation
kaggle kernels output yegorby/bp-canary-sk-cs-validation/1 -p /tmp/bp-canary-validation-20261103-v1 --file-pattern '^canary-validation.json$'
python3 scripts/gpu_bench/kaggle_canary/rescore.py /tmp/bp-kaggle-auth-20261103/results.json
python3 -m unittest discover -s scripts/gpu_bench/kaggle_canary -p 'test_*.py' -v
```

Raw files stay local in the two `/tmp` folders above. Preserve them before an OS
cleanup if further audit is needed. Checked-in receipts exclude raw transcripts
and audio; they preserve hashes and numeric results.

## Handler report changes

`reports/index-report.html`: shorter Slovak overview, directional route table,
new Kaggle evidence section, latency adjacent to results, historical detail in
native disclosures, original 34-model explorer and theme toggle retained.
Removed unsupported Czech latency, bidirectional verification, competitor prices,
GDPR assurance, Kokoro cloning claims; OmniVoice is batch/QC in the current plan.
Keyboard filter buttons, search label, live result count, Enter/Space disclosures,
focus styles, reduced motion and responsive tables added.

## Verification and limits

- `make test`: 38 passed, 3 warnings, 20.35s (existing dirty checkout).
- `.venv/bin/python scripts/bp.py selftest`: PASSED with 11 warnings. Its TTS
  instance uses the Czech voice, so this is not proof of Slovak voice quality.
- `python3 ~/.claude/scripts/config_selftest.py`: 19/19 PASS.
- New scoring/acceptance unittest: 6 passed, no model dependency/download.
- Independent reviewer recomputed all ASR scores and checked the raw result hash.
- BrowserOS desktop/mobile render and interaction checks recorded during session;
  a 390px iframe supplies a real narrow viewport, not screenshot scaling.
- Existing unrelated working-tree changes are excluded from the delivery branch.
- Canary is not integrated into live runtime; wider language and live microphone
  validation remain untested. Do not label all open-source SK ASR fundamentally
  broken based on these short clips.

## Credential incident

A pre-existing untracked handoff contained a token-like credential. Reading the
handoff surfaced it in tool output; the local file was redacted. That handoff is
excluded from commits. Treat the old credential as exposed and rotate it in
Kaggle settings, then replace the global env/token file securely. Rotation was
not performed because it revokes existing client access. Never print the values.

## Sources

- https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md#authentication
- https://huggingface.co/nvidia/canary-1b-v2
