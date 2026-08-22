# Simultaneous/incremental translation: real options, real ceiling (2026-08-22)

## Context

User asked to stop treating this as "translate an utterance after it's done" and investigate whether
a genuinely incremental, reordering-safe, near-real-time MT architecture is buildable — including
whether "LLM-style" predictive/speculative decoding could let translation start before the speaker
finishes a sentence, even for verb-final languages like German. Research only, no code changes.
Hardware constraint throughout: M1 Pro, 16GB RAM, **no CUDA**.

## 1. Simultaneous/incremental MT policies (wait-k, EMMA)

**wait-k** (Ma et al.) and its descendants (MoE-Waitk, adaptive wait-k) are real, published, and the
right conceptual family — but every usable implementation found is a **research checkpoint trained on
a specific benchmark pair** (mostly WMT De-En), not a general-purpose drop-in replacement for
opus-mt/NLLB. No pretrained wait-k checkpoint for EN↔SK was found. Building one means fine-tuning
from scratch on parallel EN-SK data with the wait-k training objective — a real ML project on its own,
not a config change. [Efficient Wait-k Models](https://www.semanticscholar.org/paper/Efficient-Wait-k-Models-for-Simultaneous-Machine-Elbayad-Besacier/8f9f7a0714408fb49eeb7060ab16bc1eedb219fd),
[Adaptive Policy with Wait-k](https://aclanthology.org/2023.emnlp-main.293/).

**EMMA** (Meta's policy, used in SeamlessStreaming) is the one credible non-research-toy candidate:
- Model: 2.5B params, checkpoints published in
  [facebookresearch/seamless_communication](https://github.com/facebookresearch/seamless_communication).
- Runtime: `fairseq2` ships prebuilt wheels for Apple Silicon Macs (confirmed from the repo's own
  install instructions) — so it *can* be installed here, but no CPU/MPS latency benchmark was found
  anywhere; every number Meta publishes assumes their own GPU cluster. This is genuinely unverified,
  not disqualified — same status SimulStreaming had before it got checked and rejected, except this one
  hasn't been checked yet.
- Slovak: the repo claims ~100 source languages; Slovak was not found explicitly confirmed with a
  BLEU number anywhere in Meta's own materials or third-party benchmarks searched. Coverage-by-language-list
  is not the same as verified quality — this is an open question, not a yes.
- **The honest number that matters**: Meta's own reporting states SeamlessStreaming hits
  **latency under 2 seconds by trading off a 66% drop in BLEU**
  ([EMMA overview](https://towardsdatascience.com/seamless-in-depth-walkthrough-of-metas-new-open-source-suite-of-translation-models-b3f22fd2834b/)).
  That is a severe quality collapse in exchange for speed — this is the actual shape of the tradeoff
  everyone doing simultaneous MT is fighting, not a solved problem.

**Verdict**: EMMA/SeamlessStreaming is the only real candidate for "translate before the sentence
ends" that isn't a training project from scratch. It is unverified on this hardware and its own
published number shows a steep quality cost. Worth a bounded spike (install, run one EN→SK and one
EN→DE sample, measure real latency and eyeball quality) before deciding anything — not worth committing
architecture around yet.

## 2. End-to-end streaming speech-to-speech models (replace STT→MT→TTS entirely)

**StreamSpeech** ([ictnlp/StreamSpeech](https://github.com/ictnlp/StreamSpeech), MIT, ACL 2024):
covers **En↔Fr/Es/De only — no Slovak**. Also uses a fixed unit-based HiFi-GAN vocoder, not voice
cloning — adopting it would throw away the already-verified, working OpenVoice cloning pipeline
(`documentation/hybrid_tts_openvoice_2026-08-21.md`) and replace your own cloned voice with a generic
synthetic one. Disqualified twice over: wrong language, wrong output identity.

**Hibiki** ([kyutai-labs/hibiki](https://github.com/kyutai-labs/hibiki), MIT/Apache-2.0/CC-BY-4.0):
**French→English only**, wrong direction entirely for this project. Notable anyway: 1B/2B params,
explicitly runs on-device via MLX on Apple Silicon, and *does* real voice transfer via
classifier-free guidance. It's proof the "compact on-device streaming S2ST model with voice
preservation" pattern is achievable at this model size — just not a model that exists for EN↔SK/DE
today. If a Hibiki-shaped model ever ships for those pairs, it's worth revisiting.

**Verdict**: no existing end-to-end model covers this project's actual language pairs without either
losing Slovak or losing the cloned voice. Not a near-term option.

## 3. "LLM-style" predictive/speculative translation

Speculative decoding is real and does deliver 2-3x LLM token-generation speedups
([NVIDIA overview](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/)),
and translation is cited as a favorable task for it (predictable continuations, high draft-acceptance
rate). The closest applied precedent found is
[Self-Speculative Biased Decoding for Faster Re-Translation](https://arxiv.org/pdf/2509.21740)
(2025), reporting **1.3-2.5x speedup** over baseline incremental re-translation. No CUDA-free
benchmark, no confirmed German/Slavic pair, and the paper's own setup assumes GPU-accelerated
decoding — running this on an M1 Pro CPU/MPS is unverified and would likely need quantization work
before it's even testable.

**The more important finding is conceptual, not a benchmark**: speculative decoding speeds up
*generating tokens the model is already confident about*. It does not solve the problem the user
raised about German specifically — **verb-final reordering is an information-availability problem,
not a compute-speed problem.** "Ich habe das Buch, das mir mein Vater letztes Jahr geschenkt hat,
gelesen" needs the verb "gelesen" (read) to correctly place the English verb early — no amount of
faster token generation lets the system emit a syntactically correct translation before the source
verb has actually been spoken, because the information doesn't exist yet. This is exactly why
`SSBD` (this project's earlier MT-stage speculative decoding test) came back negative
(6-23% slower, see prior session findings) and why EMMA's own answer to reordering-heavy pairs is to
literally return *wrong* translations 66% more often in exchange for not waiting — there is no
technique found anywhere in this research pass, LLM-based or otherwise, that removes the wait for
reordering languages without accepting a real accuracy cost. This should be stated to the user
plainly: "predict harder" is not a lever that exists here for German-style reordering. The only real
levers are (a) accept some wrong/revised output for speed (EMMA's tradeoff), or (b) wait for enough
of the clause to be safe, which is what full-sentence MT already effectively does.

## 4. Real production ceiling — what the best-funded teams actually ship

Google's Gemini 3.5 Live Translate (Google Meet, 2025) is the most relevant comparison: it is a
**native speech-to-speech model with no intermediate text stage** (removing exactly the STT→MT→TTS
chain this project uses), running on Google's own infrastructure, closed-source. Google's own
September 2025 engineering writeup states **2-3 seconds was their deliberately chosen target** —
not a compromise forced by hardware, a design decision: "faster became hard to follow; slower broke
the rhythm of conversation"
([source](https://www.startuphub.ai/ai-news/ai-research/2026/google-rolls-out-gemini-3-5-live-translate)).
By contrast, the old three-stage chain (the architecture this project also uses) is cited in the same
reporting as historically producing **10-20 seconds** of latency before recent optimization work.

**This is the grounded ceiling to communicate**: 2-3 seconds end-to-end is what the state of the art
*aims for*, deliberately, with a purpose-built native model and unlimited compute. It is not a
number this project should expect to beat with an open-source STT→MT→TTS chain on a single M1 Pro.
It is a legitimate target to aim toward once the architectural dead time (the blocking pipeline bug,
already diagnosed) is removed.

## 5. Interruption / barge-in — standard, well-precedented, not a research question

This part has an established industry pattern, confirmed across multiple current sources
([Hamming AI barge-in runbook](https://hamming.ai/resources/voice-agent-interruption-handling-runbook),
[LiveKit sequential pipeline](https://livekit.com/blog/sequential-pipeline-architecture-voice-agents)):
VAD detects new speech while TTS is playing → fire a cancel event → flush the generation queue and
playback buffer → start a fresh STT pass. The concrete implementation detail worth keeping: **TTS
output should be chunked small (100-200ms)** specifically so a cancel doesn't cut off mid-word and so
cancellation latency stays low (cited target: TTS flush under 60ms). This is directly compatible with
this project's already-working Hybrid TTS (Piper synthesis is fast enough, RTF 0.15-0.17, to chunk
without a redesign) — this is an engineering task, not a research gap.

## Proven / assumed / unknown

**Proven** (from published sources, not independently re-benchmarked here): SeamlessStreaming/EMMA
is real, ships checkpoints, and Meta's own number is <2s latency at a 66% BLEU cost. Google's shipped
system targets 2-3s deliberately with a model this project cannot replicate. Barge-in is a solved,
standard pattern. Speculative decoding is a real, orthogonal speedup technique that does not address
reordering-language latency.

**Assumed, not verified on this hardware — the real open questions for a next spike**: whether EMMA/
SeamlessStreaming actually runs at a usable speed on M1 Pro CPU/MPS at all (no benchmark exists
anywhere for this); whether its Slovak output is usable quality on real sentences (language-list
membership is not evidence); whether the 66% BLEU hit is survivable for this project's "communication
should sound flawless" bar (probably not, but should be heard, not assumed).

**Unknown/not investigated further, correctly out of scope this pass**: fine-tuning a wait-k model
from scratch for EN↔SK (a full ML training project, not a spike); whether a quantized small
decoder-only LLM could serve as an EMMA-style read/write policy controller cheaper than the 2.5B
model (no precedent found for this specific framing).

## Recommendation

1. **Do not build a from-scratch simultaneous-MT system.** No component investigated here is a safe
   bet without a real spike, and two entire candidate classes (end-to-end S2ST models, wait-k
   fine-tuning) are disqualified or out of scope respectively.
2. **The one candidate worth a bounded, time-boxed spike**: install `fairseq2` + SeamlessStreaming,
   run 5-10 real EN→SK and EN→DE sentences, measure actual latency on this M1 Pro and listen to
   actual quality against the 66%-BLEU-hit warning. Treat this exactly like the SimulStreaming
   investigation — go in ready to reject it on real evidence, don't force it to look good.
3. **Everything else this project needs for "feels near-real-time" is already engineering, not
   research**: kill the blocking pipeline bug, wire the already-verified streaming STT (T035), chunk
   TTS output, add the standard VAD-triggered barge-in/cancel pattern. None of this needs a new model.
4. **Set the user's expectation against 2-3s (Google's own deliberate target), not against zero.**
   For German-style reordering specifically, tell them plainly: there is no prediction technique that
   removes the need to wait for enough of the clause — only techniques that trade accuracy for speed.
