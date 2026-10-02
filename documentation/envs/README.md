# Side virtualenvs (deleted 2026-10-02 to lighten the project; recreate only if you rerun that experiment)

The runtime (`make run`, `make test`) only needs `.venv` from `python3.11 scripts/setup.py --dev`. These four were isolated
experiment envs (~4.7 GB together). Their `pip freeze` is saved here; recreate with:

    python3.11 -m venv .venv-<name> && .venv-<name>/bin/pip install -r documentation/envs/venv-<name>.freeze.txt

| env | used by | what it was for |
|---|---|---|
| `.venv-omni` | `scripts/bulk_hq.py`, `scripts/engine_ab.py` | OmniVoice on Apple MPS (corpus generation, engine A/B). On a GPU box just `pip install omnivoice` |
| `.venv-eval` | `scripts/bench_new_models.py`, `bench_nemotron.py` (+ `requirements-eval.txt`) | Seamless / Nemotron / zipformer / Chatterbox spikes |
| `.venv-stt` | `scripts/conversation_sim.py`, `machine_listen_qc.py`, `pipeline_latency_probe.py` | Parakeet-TDT STT subprocess |
| `.venv-train` | `scripts/finetune_personal_voice.py` | Piper fine-tune stack (`piper-tts[train]`); only for CPU-only machines, see `piper_training_runbook.md` |

Torch/CUDA wheels in the freezes are the macOS ones; on Linux/CUDA let pip resolve torch instead of the frozen build.
