# GPU bench (X-Voice + OmniVoice)

Purpose: measure synthesis RTF on any CUDA GPU (free Colab/Kaggle T4 is enough) to turn the
"a GPU would make live cloned voice possible" claim from PROJECTED into MEASURED.

1. Colab → upload `gpu_bench.ipynb` → Runtime: GPU → run cells; upload
   `processed/gpu_bench/gpu_bench_upload.zip` (script + 3 reference clips of the owner, 5–7 s each;
   EN/SK/CS). Local-only file: `*.wav` is gitignored; never commit the refs.
2. Download `results.zip` → `gpu_bench.json` + 6 wavs. Listen to the wavs (ear grade is the real metric).
3. Paste `gpu_bench.json` numbers into `documentation/model_landscape_2026-10.md` §10.1 as MEASURED
   with the GPU name.

Mac baseline (M1 Pro, MPS fp32, 2026-10-02): X-Voice marginal RTF 2.9 (CS) – 4.2 (SK);
OmniVoice RTF 0.80 on a 17 s text, 2.7 on a 1 s text (fixed overhead dominates short utterances —
relevant for live sentence-by-sentence use). Vendor: X-Voice 0.073 on RTX 4090 (UNMEASURED here).

Also runnable locally: `python gpu_bench.py run --xvoice-dir <X-Voice clone> --refs refs --out out`.
