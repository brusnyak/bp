# Canary validation on Kaggle

Use the existing Kaggle CLI; no local GPU or model install is required.
Authentication: `~/.kaggle/access_token` (mode 0600), or one command at a time:

```sh
~/.claude/scripts/with-env KAGGLE_API_TOKEN -- kaggle kernels list --user yegorby
```

`~/.env` is not automatically exported. Kaggle recognizes `KAGGLE_API_TOKEN`, not
`kaggle_api_token`. A personal token belongs in `access_token`, not the legacy
username/key `kaggle.json`. Never print auth config/token files or commit them.

```sh
kaggle kernels status yegorby/bp-canary-sk-cs-validation
kaggle kernels push -p scripts/gpu_bench/kaggle_canary --accelerator NvidiaTeslaT4 --timeout 1200
kaggle kernels output yegorby/bp-canary-sk-cs-validation -p /tmp/bp-canary-output --file-pattern '^canary-validation.json$'
python3 scripts/gpu_bench/kaggle_canary/rescore.py /tmp/bp-kaggle-auth-20261103/results.json
```

Push executes a private remote GPU job and installs dependencies/downloads the
model remotely. Obtain authorization for each additional run. Only the existing
private dataset is attached; no new recordings are uploaded. The job records
checkpoint revision, dependency versions, clip/reference hashes and first-run
inference latency (not a warmed benchmark).

ASR compares transcription with the source-language reference and divides edit
distance by **reference** word count. AST translation is saved without a score
until a reviewed target-language reference exists. A successful short-clip smoke
never verifies a full STT→MT→TTS route, real-mic WER, or voice identity. Review
reference completeness before interpreting even a short-clip score.

The 8 GiB guard limits the PyTorch CUDA allocator. It is not a measured total
process VRAM cap: driver/non-PyTorch allocations may exceed it. Peak allocated
and reserved values are separately recorded. Missing clips are recorded, not
silently substituted with another language's audio. Inspect every row, not just
the Kaggle COMPLETE state. No runtime promotion occurs here.

Historical v6 had reversed WER normalization: saved SK strings produce 3 edits /
11 reference words = 0.2727, rather than 3 / 14 hypothesis words = 0.2143.
`rescore.py` verifies this without rerunning inference. Raw outputs remain local;
the checked-in evidence receipt contains counts, scores and a source SHA-256.
