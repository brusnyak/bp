"""
SSBD (Self-Speculative Biased Decoding) wrapper around CTranslate2MT.

Paper: "Self-Speculative Biased Decoding for Faster Re-Translation" (arXiv 2509.21740).
Reference speedups reported in the paper (Flores dataset, Tower+ 2B, beta=0.2):
en->de 1.69x, en->zh 1.48x, en->ja 1.36x, with comparable COMET quality.
Those numbers are from a different model/dataset and are NOT assumed to transfer here
-- see benchmark_ssbd_mt.py for real numbers measured on this project's Opus-MT models.

Adaptation note (read before trusting this as a literal paper reimplementation):
The paper's "lightweight bias" is a per-position mixture
P'(y_i) = (1-beta)*P(y_i) + beta*delta(y_i, draft_i), with divergence at the first
position where argmax(P') != draft token. CTranslate2's public Python API does not
expose the full per-step output distribution during a teacher-forced pass, so the
literal argmax-of-mixture check is not implementable without patching the C++ decoder.
This implementation instead uses CTranslate2's score_batch() -- a genuine single
parallel forward pass, teacher-forced, over the previous draft against the NEW source
-- to get a per-token log-probability, and treats "log-prob of the draft token drops
below a threshold" as the divergence signal (a confidence-threshold proxy for the
paper's biased-argmax-mismatch criterion). This preserves the actual mechanism that
produces the speedup (parallel verification of the unchanged prefix instead of
autoregressive regeneration of it) even though the acceptance criterion is not
bit-identical to the paper's formula. CTranslate2 does separately expose a native
`prefix_bias_beta` option on translate_batch, which is the same bias-toward-prefix
idea applied during generation itself; it is not used here because it does not by
itself skip recomputation of the accepted prefix -- decoding still proceeds token by
token even when biased toward the prefix, which is why the score_batch + target_prefix
combination below is used for the actual accept/resume step instead.
"""

from typing import List, Optional, Tuple

from backend.mt.ctranslate2_mt import CTranslate2MT


class SSBDCTranslate2MT(CTranslate2MT):
    def __init__(self, model_path: str = "Helsinki-NLP/opus-mt-en-sk", device: str = "auto",
                 accept_log_prob_threshold: float = -1.5):
        super().__init__(model_path=model_path, device=device)
        # Threshold in log-prob space (natural log). -1.5 ~= accept tokens the
        # model still assigns >~22% probability to under the new source context.
        # This is a tunable hyperparameter, not a value derived from the paper
        # (the paper tunes beta instead, which this implementation cannot use
        # directly -- see module docstring).
        self.accept_log_prob_threshold = accept_log_prob_threshold

    def _tokenize(self, text: str) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(
            self.tokenizer.encode(text, add_special_tokens=True)
        )

    def _detokenize(self, tokens: List[str]) -> str:
        return self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True
        )

    def translate_streaming(
        self,
        text: str,
        prev_draft_tokens: Optional[List[str]] = None,
        prev_draft_complete: bool = False,
        beam_size: int = 4,
    ) -> Tuple[str, List[str], bool, float, int, int]:
        """
        Translate `text` (the current, possibly-grown source segment), optionally
        reusing `prev_draft_tokens` (the target-token output from the previous,
        shorter version of this segment) as a speculative draft.

        `prev_draft_complete` must be True only if the previous draft actually
        ended on the model's end-of-sequence token (see `is_complete` in the
        return tuple) -- CTranslate2's translate_batch strips the end token from
        `hypotheses` by default, so completion is NOT the same thing as "the
        draft has more than zero tokens" and must be tracked explicitly. Skipping
        this check was a real bug in an earlier version of this function: a short,
        genuinely-finished draft (e.g. "I think" -> "Myslim,") was being silently
        reused verbatim for much longer follow-on source text, because verification
        only checks "does the model still agree with these tokens", not "is this
        actually the whole translation" -- an incomplete-but-still-locally-plausible
        draft was passing verification and never being extended.

        Returns: (translated_text, new_draft_tokens, is_complete,
                   wall_clock_seconds, accepted_prefix_len, prev_draft_len)
        """
        import time

        source_tokens = self._tokenize(text)
        start = time.perf_counter()

        if not prev_draft_tokens:
            results = self.translator.translate_batch(
                [source_tokens], max_batch_size=1, beam_size=beam_size,
                num_hypotheses=1, return_end_token=True,
            )
            new_tokens = results[0].hypotheses[0]
            is_complete = bool(new_tokens) and new_tokens[-1] == "</s>"
            elapsed = time.perf_counter() - start
            out_tokens = new_tokens[:-1] if is_complete else new_tokens
            return self._detokenize(out_tokens), new_tokens, is_complete, elapsed, 0, 0

        # Step 1: verify the previous draft against the NEW (grown) source in a
        # single parallel teacher-forced forward pass.
        scoring = self.translator.score_batch([source_tokens], [prev_draft_tokens])
        log_probs = scoring[0].log_probs

        accept_len = 0
        for lp in log_probs:
            if lp < self.accept_log_prob_threshold:
                break
            accept_len += 1

        accepted_prefix = prev_draft_tokens[:accept_len]
        fully_accepted = accept_len == len(prev_draft_tokens)

        # Step 2: only skip new decoding entirely if the WHOLE draft was verified
        # AND it was already a genuinely complete translation (ended on EOS).
        # Otherwise always resume decoding (cheap: forced prefix means only the
        # new tail needs autoregressive generation, not the whole sequence).
        if fully_accepted and prev_draft_complete:
            elapsed = time.perf_counter() - start
            return self._detokenize(accepted_prefix), accepted_prefix, True, elapsed, accept_len, len(prev_draft_tokens)

        results = self.translator.translate_batch(
            [source_tokens],
            target_prefix=[accepted_prefix] if accepted_prefix else None,
            max_batch_size=1,
            beam_size=beam_size,
            num_hypotheses=1,
            return_end_token=True,
        )
        new_tokens = results[0].hypotheses[0]
        is_complete = bool(new_tokens) and new_tokens[-1] == "</s>"
        elapsed = time.perf_counter() - start
        out_tokens = new_tokens[:-1] if is_complete else new_tokens
        return self._detokenize(out_tokens), new_tokens, is_complete, elapsed, accept_len, len(prev_draft_tokens)
