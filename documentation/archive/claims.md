# Claims ledger (public pages)

Rule: a claim on the site is either proven (evidence file) or a target. While a claim is a target, the page says so.
Updated 2026-11-05. Source: documentation/implementation-2026-11-04/*, README.md, AGENTS.md, PRODUCT.md.

| # | Claim | Status | Evidence now | Test that closes it | Page copy until then |
|---|-------|--------|--------------|---------------------|----------------------|
| 1 | English to Slovak live translation works | proven, small sample | streaming-findings.md, one speaker, 45 s and 90 s inputs | 20+ sentences graded by native Slovak listeners | "Works; quality not yet scored" |
| 2 | Slovak to English works | partial | Slovak recognition WER 0.27-0.32 vs the 0.2 gate (AGENTS.md); replay WER 0.39-0.71 (vault review 2026-11-02) | Frozen bench in `bench/` with kinit large-v3-turbo-sk on a T4; target WER <= 0.2 | "Works in both directions; Slovak recognition is the weaker half" |
| 3 | Speech in the speaker's own voice | proven for the author's Slovak voice, identity ungraded; any-user cloning is research | en-sk.json personal Piper voice (used in the M1 delay runs); OmniVoice T4 runs | Bilingual listener grades (identity + intelligibility), then OmniVoice on target hardware | "Own-voice Slovak for the author; cloning anyone's voice is research" |
| 4 | Near real time: mean 2.5 s per word, 0.6-0.8 s to first sound | proven, one speaker, one Mac | streaming-findings.md lines 41-62 | 10-minute continuous rehearsal, 3 speakers, p50/p95 speech-end to first audio, max queue delay | Show the numbers with "ear-voice span of human interpreters is about 2-4 s (unverified)" |
| 5 | Alternating English and Slovak in one session | unproven; T4 load test shows 8-11 s backlog | load_tests_2026-10.md | Add a swap-direction control, then a 10-minute alternating run | "Switch direction between turns" without latency claim |
| 6 | Many simultaneous sessions | unknown | none | Concurrent-session load test (1, 2, 4, 8) on intended hardware | "One session measured" |
| 7 | Works in a hybrid meeting room (LHC) | unknown | PRODUCT.md integration hypothesis only | Rehearsal on the real room audio path | "Prototype; room integration not tested" |
| 8 | Personal data stays local | proven by code for audio; recordings stored on the server | privacy.html, backend/main.py | Retention and deletion test | as in the privacy page |
