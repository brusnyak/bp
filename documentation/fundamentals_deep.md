# Fundamentals, worked deep (BP companion to `fundamentals.md`)

Every section ends with a number you can verify in 2 minutes with system python3.
No frameworks needed — only the math our pipeline actually runs.

## 1. One backprop step, by hand

Neuron: `z = w·x + b`, `ŷ = σ(z)`, loss `L = (ŷ − y)²`. Take `x=2, w=0.5, b=0, y=1`:

- `z = 1.0`, `ŷ = σ(1) ≈ 0.731`, `L ≈ 0.0723`
- `dL/dŷ = 2(0.731−1) = −0.538`; `dŷ/dz = 0.731·(1−0.731) ≈ 0.197`
- `dL/dw = −0.538 · 0.197 · 2 ≈ −0.212` → Adam/SGD nudges `w` up by `η·0.212`
- Verify: `python3 -c "import math;s=1/(1+math.exp(-1));print(round(2*(s-1)*s*(1-s)*2,3))"` → `-0.212`

That product of three local derivatives IS backprop. A 1B-param model does exactly
this, 1B times, in parallel. Nothing else is going on.

## 2. One attention head, by hand

Tokens as rows, `d=2`. Inputs X = [[1,0],[0,1]] ("two one-hot words"), single head with
identity projections (Q=K=V=X) and no scaling, to keep arithmetic visible:

- Scores `Q·Kᵀ` = [[1,0],[0,1]] (each word matches itself, ignores the other)
- `softmax` per row → [[0.731,0.269],[0.269,0.731]]
- Output = weights·V = [[0.731,0.269],[0.269,0.731]] — each word now carries 27% of
  the other. That blending is "context". Real heads learn Q,K,V matrices that make
  the blend linguistically meaningful (subject↔verb, pronoun↔antecedent).
- Verify: `python3 -c "import math;e=math.exp(1);a=e/(e+1);print(round(a,3),round(1-a,3))"` → `0.731 0.269`
- Scaling `1/√d` exists because without it, large dot products saturate softmax and
  gradients die (same sigmoid-saturation disease as §1, one level up).
- Multi-head = several such blends in parallel, concatenated. Whisper-small: 12 heads ×
  12 layers encoder + 12 × 12 decoder. O(n²): doubling audio length quadruples attention
  memory — why 30s windows and chunking exist.

## 3. Beam search, walked through

Vocab {A,B}, beam=2, model scores for step 1: P(A)=0.6, P(B)=0.4. Keep both.
Step 2 given each: from A: P(A)=0.5,P(B)=0.5; from B: P(A)=0.9,P(B)=0.1. Candidates:
AA=0.30, AB=0.30, BA=0.36, BB=0.04 → keep BA, AA (or AB tie). Greedy (beam=1) would
commit to A then tie-break blindly and can never reach BA=0.36, the global best.
That gap — greedy lock-in vs beam recovery — is what our `beam_size` benchmark on
`small-sk` will price: 30–50% slower for how much WER?

## 4. CTC (why SenseVoice is fast)

Speech frames outnumber transcript characters ~10:1 and alignment is unknown. CTC
allows each frame to emit a character or BLANK, then collapses (`hhee__lllo` → `helo`).
Loss = sum over ALL valid alignments (dynamic programming, forward-backward — same
algorithm family as HMMs). No autoregression: one parallel pass. Price: conditional
independence per frame (can't model "q→u" spelling rules internally) → slightly worse
than autoregressive on hard audio. This is the exact speed/quality trade of the
SenseVoice row in our candidate list.

## 5. VITS / Piper (three losses, one voice)

- **VAE + monotonic alignment**: text (phonemes) and audio (mel) are different lengths;
  Monotonic Alignment Search finds the cheapest left-to-right pairing (dynamic
  programming again). The encoder learns a distribution over sounds per phoneme —
  sampling it with different noise = same text, slightly different delivery.
- **Normalizing flows**: a chain of invertible maps turning simple noise into complex
  speech distributions (mathematically exact likelihood — no approximation).
  Fine-tuning re-tunes these maps + the duration predictor toward new data. 18 clips
  moved ours only halfway (F0 range halved = durations/timbre collapsed inward).
- **Adversarial (GAN) vocoder**: a discriminator net tries to spot fake waveforms;
  the generator improves until it can't. This is where "natural vs robotic" is won or
  lost — and why our undertraining shows as flat prosody, not wrong words.
- Duration predictor footnote: it decides how many frames each phoneme gets — i.e.
  speaking rate. Our personal voice's ~50%-slower pace lives here, learned from 18
  slow careful reading clips. Training data pace = output pace. Choose corpus pace
  deliberately (our bulk corpus reads at natural meeting pace — good).

## 6. Diffusion / flow-matching (OmniVoice's engine)

Forward: `x_t = √(ᾱ_t)·x_0 + √(1−ᾱ_t)·ε`, ε ~ N(0,1) — picture fading into static over
t=0→T. The net learns to predict the noise ε (or velocity) at each t. Reverse: start
from pure noise, subtract predicted noise T times → speech. Fewer steps (our
num_step=16 vs default 32) = coarser denoising = faster but rougher — the exact knob
behind Supertonic's 2-step-robotic / 5-step-natural row. Worked 1D check:
`x_0=5, ᾱ=0.5, ε=1` → `x_t = 0.707·5 + 0.707·1 ≈ 4.24`; the net's whole job is
recovering that 5 from 4.24, repeatedly, in 24kHz×80-mel dimensions.

## 7. Mel scale + quantization, numerically

- Mel: `m = 2595·log10(1 + f/700)`. 1000 Hz → 1000 mel (by design); 8000 Hz → 2840 mel
  (compressed — matches hearing). 80 mel bins × 16kHz audio is Whisper's retina.
- INT8 quant: `q = round(x/s) + z`, `s = (max−min)/255`. Weights in [−1,1] → s≈0.0078,
  max rounding error ±0.0039 per weight — invisible after 100M of them vote. Activations
  (outliers!) hurt more than weights — why CTranslate2 int8 keeps accuracy but naive
  activation quant doesn't. Our 0.1s MT runs on exactly this.

## 8. Reading our numbers with these eyes

- STT 0.51s on 5s audio (base, int8, CoreML-ish CPU): autoregressive decode of ~15
  tokens ≈ 30ms/token. small-sk slower = bigger decoder, same game.
- MT 0.1s: tiny encoder-decoder, int8, short sequences — attention O(n²) is trivial here.
- Piper RTF 0.04: single forward pass, no autoregression, no diffusion steps.
- Omni RTF ~1.0: 16 reverse-diffusion steps × transformer cost. MPS barely helps (2.9→2.5)
  because the cost is sequential steps, not parallel math — Amdahl's law wearing a robe.
- VAD 0.3s close rule vs synth-bed noise: a 20-cent classifier losing to a constant
  stimulus. Energy-gating (our queued fix) is just a second, dumber vote on top.
