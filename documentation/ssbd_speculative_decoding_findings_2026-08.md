# SSBD (Self-Speculative Biased Decoding) — findings, 2026-08

Source paper: "Self-Speculative Biased Decoding for Faster Re-Translation," arXiv 2509.21740.
Implementation: `backend/mt/ssbd_ctranslate2_mt.py`. Benchmark: `benchmark_ssbd_mt.py`.

## What the paper claims

For simultaneous/streaming translation, where the target is repeatedly regenerated as
the source grows, SSBD reuses the previous output as a speculative draft, verifies it
against the new source in a single forward pass with a bias term
`P'(y_i) = (1-beta)*P(y_i) + beta*delta(y_i, draft_i)`, and resumes normal
autoregressive decoding only from the first position where the draft disagrees with
the (biased) model. Reported speedups (Flores dataset, Tower+ 2B model, beta=0.2):
1.36x-1.69x, with COMET quality within 0.002 of the non-speculative baseline.

## Adaptation for this project

CTranslate2's public Python API does not expose per-step output distributions during
a teacher-forced pass, so the literal argmax-of-biased-mixture divergence check from
the paper could not be implemented without patching CTranslate2's C++ decoder. The
implementation instead uses:
- `Translator.score_batch(source, prev_draft)` — a genuine single parallel
  teacher-forced forward pass — to get a per-token log-probability of the previous
  draft under the *new* source.
- A log-probability threshold (`accept_log_prob_threshold`, default -1.5 nats) as the
  divergence signal, in place of the paper's biased-argmax-mismatch criterion.
- `Translator.translate_batch(target_prefix=accepted_prefix)` to resume decoding only
  from the first rejected position.

This preserves the mechanism that produces the paper's speedup (skip autoregressive
decode of an unchanged prefix via a cheap parallel verification pass) even though the
acceptance criterion is not bit-identical to the paper's formula.

## A real bug found and fixed during implementation

CTranslate2's `translate_batch` strips the end-of-sequence token from returned
hypotheses by default. An early version of this implementation treated "the whole
previous draft scored above threshold" as "the translation is complete," and returned
it verbatim with zero new decoding. This is wrong: a short draft (e.g. "I think" ->
"Myslim,") can keep scoring plausibly under score_batch even as the source grows to
a full sentence, without that meaning the draft is a complete translation of the new,
longer source. Result: the SSBD path silently stopped extending the translation and
returned a stale fragment on every subsequent increment, while reporting an (invalid)
4.83x speedup because it was doing far less work than the baseline — not equivalent
work, faster. Fixed by requiring `return_end_token=True` and only skipping new
decoding when the accepted prefix both matches in full *and* actually ends on EOS.
Flagging this prominently because it is exactly the kind of bug that silently produces
an impressive-looking but meaningless benchmark number if not checked by hand against
the actual translated text, not just the latency figure.

## Real measured results (M1 Pro, CPU, int8, `Helsinki-NLP/opus-mt-en-sk`)

| Scenario | Beam size | Baseline total | SSBD total | Speedup | Draft reuse |
|---|---|---|---|---|---|
| Short growing utterance (5-11 tokens/step) | 4 | 305.1ms | 398.7ms | **0.77x** | 0/0, 3/4, 7/8, 11/10, 8/11 |
| Long growing utterance (conference-length) | 4 | 565.2ms | 603.1ms | **0.94x** | 0/0, 0/9, 0/18, 0/15, 0/18 |
| Long growing utterance (conference-length) | 1 (greedy) | 352.5ms | 378.2ms | **0.93x** | 0/0, 0/9, 0/10, 1/21, 0/25 |

**SSBD is slower than the baseline in every configuration tested — not faster.**

## Root cause

Draft reuse (how many previous-draft tokens survived verification) is near zero for
the long-utterance scenario in both beam=4 and beam=1 runs — 0 of 9-25 tokens reused
on 4 of 5 increments. Inspecting the actual translations shows why: as more English
source context arrives, `Helsinki-NLP/opus-mt-en-sk` frequently **restructures the
whole Slovak sentence** (different word order, different clause placement) rather
than appending to its previous output. Example, same underlying utterance:

- After "...today": `Takže hlavná vec, ktorú chcem dnes povedať` ("So the main thing I want to say today")
- After "...today is that our current approach to the": `Hlavnou myšlienkou, ktorú dnes chcem povedať, je, že náš súčasný prístup k` (restructured to "The main idea I want to say today is that our current approach to")

This held under both beam search (beam=4) and greedy decoding (beam=1), ruling out
beam-search re-ranking as the cause. SSBD's core assumption — that the previous
translation is usually a valid, reusable prefix of the next one — does not hold for
this model on this data.

## Proven / assumed / unknown

- **Proven** (measured directly, this section's table): SSBD as implemented here is
  6-23% slower than the baseline for `Helsinki-NLP/opus-mt-en-sk`, in three different
  configurations.
- **Proven** (inspected directly): the failure mode is near-zero draft-token reuse
  caused by whole-sentence restructuring as context grows, not an implementation
  inefficiency in the verification/resume mechanics themselves.
- **Assumed, not tested**: that this is a property of general-purpose sentence-level
  NMT models (trained on complete sentence pairs, no notion of incremental
  re-translation) rather than a quirk specific to this one Opus-MT checkpoint. Plausible
  given the mechanism (nothing in training encourages prefix-stability across growing
  inputs), but not verified against a second MT model.
- **Unknown**: whether a model actually trained/fine-tuned for incremental streaming
  re-translation — closer to the paper's own Tower+ 2B setup — would show the
  prefix-stable behavior SSBD depends on. Not tested; would require either training
  such a model or finding an existing open one, out of scope for this pass.

## Recommendation

Do not adopt SSBD in the shipped pipeline as currently implemented — it makes latency
worse, measured, not assumed. Keep the implementation and this write-up as a
legitimate negative-result thesis contribution: a plausible-sounding published
technique was implemented faithfully (adapted only where the target library's public
API required it), tested honestly including a real bug that would have produced a
fake positive result if unchecked, and found not to transfer to this project's actual
model class, with a specific, inspectable reason why. If real-time perceived-latency
work continues on this project, the more promising direction (per the same session's
separate research) is the TTS-stage voice-cloning latency fix (Piper fine-tuning /
voice conversion), not MT-stage speculative decoding.
