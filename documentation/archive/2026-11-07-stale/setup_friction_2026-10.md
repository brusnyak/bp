# First-setup friction, measured on a fresh clone — 2026-10-02 (M1 Pro Mac, cold HF + pip caches)

Method: `git clone https://github.com/brusnyak/bp.git` into an empty dir, `HF_HOME`/`PIP_CACHE_DIR` pointed at empty dirs,
`python3.11 scripts/setup.py --dev`, then `make test`, `make run`, `make demo-check`. Nothing from the owner's machine reused.

| step | time | note |
|---|---|---|
| `git clone` | **89 s** | **705 MB** `.git` for an 18 MB working tree: old commits carried model binaries (opus-mt, Piper onnx). Fixable only by rewriting history of the public repo (not done) |
| `setup.py --dev` (venv, deps, .env, cert, Piper base voices, Whisper base, throwaway torch env, CTranslate2 conversion of 3 MT + `whisper-small-sk`) | **428 s** | ends with 656 MB `.venv`, 618 MB `ct2_models`; the conversion step downloads ~1.8 GB into the HF cache (safe to delete afterwards) and a ~0.8 GB torch env that is removed |
| `make test` | 22 s (first run 68 s incl. imports) | 38/38 after the fixes below |
| `make run` → HTTPS up | ~6 s | self-signed cert warning in the browser |
| total, clone → green tests → server up | **≈ 9 min** | plus a one-time browser cert click |

## Friction found and fixed in this pass
1. **3 of 38 tests failed on a fresh clone** (ratings API): they needed `processed/ear_grades.json`, which is generated + gitignored. `setup.py` now builds it (`scripts/grade_library.py --apply`) and the test fixture builds/removes it itself.
2. **Dead `npm install` step**: chart.js is vendored in `ui/vendor/`; `node_modules` was never referenced. It also dirtied `package-lock.json` in the fresh clone (blocking `git pull`). Step, `package.json`, `package-lock.json` removed (`--no-npm` flag kept as a no-op for old CI lines).
3. **`make demo-check` reported REQUIRED FAILED on any clone**: it demanded the owner's personal voices, speaker registry, Google client id, OpenVoice. On a fresh clone (no personal SK voice) those are now warnings; the app uses the generic Piper voice.
4. Removed locally: four side venvs (4.7 GB; `pip freeze` kept in `documentation/envs/`), superseded `report/` landing.

## Left as is (decide if it bites)
- 705 MB clone (history rewrite = breaks the Windows fork's checkout; needs agreement).
- `models/openvoice_v2` (125 MB, legacy hybrid-clone path) is optional and absent in a fresh setup; the live app no longer needs it now OmniVoice is the cloning engine.
- Google login needs `GOOGLE_CLIENT_ID` in `.env` (owner secret, by design).
- No personal voice on a fresh machine: cloning needs the owner's recording (recording wizard) + a GPU for OmniVoice, or the owner's Piper voice copied by hand (backed up outside git in `~/Documents/STU/BP-voice-backup`).
