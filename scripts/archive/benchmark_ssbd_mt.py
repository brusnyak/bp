"""
Real A/B benchmark: standard re-translation-from-scratch vs SSBD-augmented MT,
on a simulated streaming scenario (a source sentence growing incrementally, the
way VAD-triggered segments actually arrive in backend/main.py's handle_audio_stream).

Run: python benchmark_ssbd_mt.py
Requires ct2_models/Helsinki-NLP--opus-mt-en-sk to already exist (it does, in this repo).
"""

import statistics
import sys

from backend.mt.ctranslate2_mt import CTranslate2MT
from backend.mt.ssbd_ctranslate2_mt import SSBDCTranslate2MT

# Simulated incremental VAD segments for one growing utterance, plus a second,
# unrelated utterance to test that a topic change doesn't wrongly reuse a draft.
STREAM = [
    "I think",
    "I think we should",
    "I think we should go",
    "I think we should go now",
    "I think we should go now before it rains",
]

# Longer, conference-speech-realistic growing utterance: the paper's speedup
# comes from skipping autoregressive decode of an unchanged prefix, so it should
# show up more (if it shows up at all on this model) as output length grows.
STREAM_LONG = [
    "So the main point I want to make today",
    "So the main point I want to make today is that our current approach to the",
    "So the main point I want to make today is that our current approach to the problem has been too narrow",
    "So the main point I want to make today is that our current approach to the problem has been too narrow and we need to consider",
    "So the main point I want to make today is that our current approach to the problem has been too narrow and we need to consider a completely different strategy going forward",
]

MODEL_PATH = "Helsinki-NLP/opus-mt-en-sk"


def run_baseline(model: CTranslate2MT, stream):
    latencies = []
    for text in stream:
        _, latency = model.translate(text, "en", "sk")
        latencies.append(latency)
    return latencies


def run_ssbd(model: SSBDCTranslate2MT, stream):
    latencies = []
    accept_ratios = []
    prev_draft = None
    prev_complete = False
    for text in stream:
        translated, prev_draft, prev_complete, latency, accepted, prev_len = model.translate_streaming(
            text, prev_draft_tokens=prev_draft, prev_draft_complete=prev_complete
        )
        latencies.append(latency)
        accept_ratios.append(f"{accepted}/{prev_len}" if prev_len else "n/a (fresh)")
        print(f"  '{text}' -> '{translated}'  ({latency*1000:.1f}ms, reused {accepted}/{prev_len} draft tokens, complete={prev_complete})")
    return latencies, accept_ratios


def run_scenario(name, stream, baseline_model, ssbd_model):
    print(f"\n########## Scenario: {name} ##########")
    print("--- Baseline: full re-translation from scratch on every increment ---")
    baseline_latencies = run_baseline(baseline_model, stream)
    for text, lat in zip(stream, baseline_latencies):
        print(f"  '{text[:60]}...'  ({lat*1000:.1f}ms)")

    print("--- SSBD: reuse previous draft, verify in one pass, resume from divergence ---")
    ssbd_latencies, accept_ratios = run_ssbd(ssbd_model, stream)

    total_baseline = sum(baseline_latencies)
    total_ssbd = sum(ssbd_latencies)
    print(f"Baseline total: {total_baseline*1000:.1f}ms  (mean {statistics.mean(baseline_latencies)*1000:.1f}ms)")
    print(f"SSBD total:     {total_ssbd*1000:.1f}ms  (mean {statistics.mean(ssbd_latencies)*1000:.1f}ms)")
    speedup = total_baseline / total_ssbd if total_ssbd > 0 else float("nan")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Draft reuse per increment: {accept_ratios}")
    return speedup


def main():
    print(f"Loading {MODEL_PATH} (int8, CPU) ...")
    baseline_model = CTranslate2MT(model_path=MODEL_PATH, device="auto")
    ssbd_model = SSBDCTranslate2MT(model_path=MODEL_PATH, device="auto")

    # Warm up (first call pays one-off model/graph warmup cost on both).
    baseline_model.translate("warm up", "en", "sk")
    ssbd_model.translate_streaming("warm up", prev_draft_tokens=None)

    speedup_short = run_scenario("short growing utterance (5-11 tokens)", STREAM, baseline_model, ssbd_model)
    speedup_long = run_scenario("longer growing utterance (conference-speech length)", STREAM_LONG, baseline_model, ssbd_model)

    print("\n=== Summary ===")
    print(f"Short-utterance speedup: {speedup_short:.2f}x")
    print(f"Long-utterance speedup:  {speedup_long:.2f}x")


if __name__ == "__main__":
    sys.exit(main())
