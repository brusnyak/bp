# BP CLI Quick Reference

> Single entry: `scripts/bp.py` (alias `bp` after `make install` or `pip install -e .`)

---

## Language Packs

```bash
bp language add <pair>           # add + smoke test (e.g. pl, de, fr)
bp language test <pair>          # run fixed clip, write WER/chrF/latency
bp language list                 # show all packs with status
bp language remove <pair>        # delete pack + local models
```

**Pack file** (`languages/packs/sk-en.json`):
```json
{
  "pair": "sk-en",
  "source": "sk", "target": "en",
  "stt": { "engine": "faster_whisper", "model": "small-sk", "language": "sk",
           "beam_size": 5, "initial_prompt": "Slovenská transkripcia s diakritikou: á, ä, č, ď, é, í, ĺ, ľ, ň, ó, ô, ŕ, š, ť, ú, ý, ž." },
  "mt": { "engine": "opus-mt", "model": "Helsinki-NLP--opus-mt-sk-en" },
  "tts": { "engine": "piper", "voice": "en_US-ryan-medium", "fallback": "en_US-lessac-medium" },
  "clip": "speaker_voices/me_sk_script.m4a",
  "license": "commercial-ok"
}
```

---

## Engine Registry

```bash
bp engine list                   # all engines: name, stage, available, licence
bp engine check <stage> <name>   # licence gate (fail closed)
  --allow-noncommercial          # internal only (XTTS, NLLB)
  --allow-unknown                # internal only (unverified model card)
bp engine gate <stage> <name>    # quality/latency gate, writes processed/engine_gate/
```

**Stages**: `vad`, `stt`, `mt`, `tts`, `unified`

**Licence policy** (fail closed):
- `commercial-ok` → ships by default
- `non-commercial` → blocked unless `--allow-noncommercial`
- `unknown` → blocked unless `--allow-unknown`

---

## Verification

```bash
bp selftest                      # hardware + core engines + pipeline + registry + packs
bp selftest --strict             # fail on any missing optional engine
make test                        # 38 pytest tests (hardware, VAD, MT, API, auth, security, config)
```

---

## Benchmarks

```bash
bp stt --models base,small --clip sk --lang-prompt sk,cs,auto
# writes processed/stt_matrix.json

bp engine gate stt faster_whisper   # STT gate (WER < 0.15, latency < 8s)
bp engine gate mt opus-mt           # MT gate (chrF > 0.6, latency < 0.5s)
bp engine gate tts piper            # TTS gate (RTF < 0.1)
bp engine gate vad webrtcvad        # VAD gate (false_start < 0.05)
```

---

## Voice Lab

```bash
bp grades                        # machine pre-grade + merge into library.json
bp grades --import voice-ratings.json  # import browser ear verdicts
bp library                       # refresh Voice Lab manifest (library.json)
bp corpus                        # normalize/dedupe/transcribe voice clips
```

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `BP_STT_BEAM_SIZE` | `5` | Whisper beam size (1 = greedy, faster) |
| `BP_SK_STT_MODEL` | unset | Override SK STT model (name or CT2 dir) |
| `BP_ENABLE_RATE_LIMIT` | unset | Enable slowapi rate limiting (`=1` to enable) |
| `BP_REGISTRATION_OPEN` | unset | Allow user registration (`=1` to enable) |
| `BP_DEMO_USER` | unset | Create test@example.com / password user |
| `BP_HOST` | `localhost` | Bind address (use `0.0.0.0` for LAN demo) |
| `GOOGLE_CLIENT_ID` | unset | Enable Google OAuth button |
| `JWT_SECRET` | required | HS256 secret for auth tokens |

---

## One-Command Setup

```bash
python3.11 scripts/setup.py --dev
# Creates .venv, installs deps, converts models, generates certs, downloads voices, runs bp selftest
```

---

## Run

```bash
make run        # https://localhost:8000 (self-signed cert in certs/)
make lab        # http://localhost:8080/ui/voice-lab/lab.html (static, no backend)
make demo-check # pre-flight: assets + live server check
```

---

## Pro Tips

- **SK→EN fast mode**: `BP_STT_BEAM_SIZE=1 make run` (WER ~0.32 vs 0.23, ~30% faster)
- **Debug audio**: `test_output/` gets segment WAVs when `backend/main.py` debug save enabled
- **Force MT conversion**: delete `ct2_models/Helsinki-NLP--opus-mt-<pair>` and re-init
- **Fresh clone on Windows**: same `setup.py --dev` works (torch-free, Piper-only)