#!/usr/bin/env python3
"""
One-time, offline personal-voice fine-tuning for the fast TTS tier (User Story 5 / T025-T027).

Deliberately NOT part of the main FastAPI backend or requirements.txt: this needs a
separate training toolchain (PyTorch Lightning, scikit-build, Cython, cmake/ninja) that
has no business being installed on a real-time inference server. Run this once per
target language to produce a `.onnx` + `.onnx.json` pair, then drop the output into
`backend/tts/piper_models/` like any other Piper voice - the serving backend never
needs to know a fine-tune happened.

Setup (once, in its own venv - NOT the project's main venv):
    python3.11 -m venv .venv-train        # piper1-gpl's training stack wants <=3.11/3.12
    .venv-train/bin/pip install piper-tts[train]
    brew install cmake ninja espeak-ng    # macOS; apt-get equivalents on Linux
    # build the monotonic_align Cython extension (see piper1-gpl/build_monotonic_align.sh -
    # the packaged wheel ships core.pyx but not the compiled .so, must build once):
    #   cd <venv>/lib/python3.11/site-packages/piper/train/vits/monotonic_align
    #   python setup.py build_ext --inplace && mkdir -p monotonic_align && \
    #     find . -name 'core*.so' -exec cp {} monotonic_align/ \\;
    # PyTorch's new (2.9+) default ONNX exporter fails on this VITS model's
    # data-dependent control flow (GuardOnDataDependentSymNode in
    # rational_quadratic_spline). Export must force the legacy TorchScript exporter
    # (torch.onnx.export(..., dynamo=False)) - handled by this script.

Verified hands-on (2026-08-19, M1 Pro, CPU, no dedicated GPU):
  - Piper's own training doc gives no CPU/MPS timing figures at all (hardware section only
    cites a Threadripper + A6000/3090 GPU setup) - had to measure this machine directly.
  - MPS was SLOWER than CPU for this model: ~0.04-0.07 it/s (MPS) vs ~0.25-0.32 it/s (CPU),
    because VITS's constant-padding ops aren't natively supported on MPS and silently fall
    back to a slow View-Ops path (visible as a UserWarning during training). Use --accelerator
    cpu, not mps, on Apple Silicon, despite what "use the GPU" defaults would suggest.
  - At the measured ~3.5s/training-step (CPU, batch_size=1), a real fine-tune run
    (order-of-magnitude estimate: 1,000-5,000 steps from a pretrained checkpoint for a
    single-speaker adaptation - this range is a reasoned estimate, not a sourced figure,
    treat it as a planning ballpark and re-measure once a real run is done) is roughly
    1-5 hours on this hardware. Budget accordingly - this is NOT something to run live.
  - Exported+inference RTF on the fine-tuned model: 0.0295 (measured, 5-run average) -
    confirms the entire point: once fine-tuned and exported to .onnx, synthesis runs at
    Piper-native speed with zero XTTS-style runtime cloning overhead. XTTS's RTF is 1.72
    (documentation/COQUI_TTS_PERFORMANCE_REPORT.md) - roughly a 58x speed difference.

Open, unresolved (flagging per Constitution Principle II - evidence-first, don't assert):
  - Minimum viable recording length: could not find an authoritative source pinning an
    exact minute count. The user proposed ~1 minute; general VITS/Piper fine-tuning
    community practice (not a single hard citation) points to needing meaningfully more
    than that for good quality. This script defaults to recommending 10-20 minutes as a
    practical starting point (low end of the already-scoped 10-60 min range in tasks.md
    T026), not 1 minute - build the recording UX around that, and treat "is 1 minute
    actually enough" as an open question to answer empirically once a real quality
    comparison is done, not something to assume either way.
"""

import argparse
import csv
import json
import subprocess
import sys
import wave
from pathlib import Path


def read_speaker_voices(speaker_voices_dir: Path, language: str):
    """Reuse the existing speaker_voices.json convention instead of building new recording infra."""
    metadata_path = speaker_voices_dir / "speaker_voices.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No speaker_voices.json at {metadata_path}")
    entries = json.loads(metadata_path.read_text())
    matched = [
        e for e in entries
        if e.get("language", "en").startswith(language) and e.get("transcribed_text")
    ]
    return matched


def build_dataset_csv(entries, speaker_voices_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metadata.csv"
    total_seconds = 0.0
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        for e in entries:
            src = speaker_voices_dir / Path(e["path"]).name if "path" in e else speaker_voices_dir / e["filename"]
            if not src.exists():
                print(f"WARNING: skipping missing file {src}", file=sys.stderr)
                continue
            with wave.open(str(src)) as w:
                total_seconds += w.getnframes() / w.getframerate()
            writer.writerow([src.name, e["transcribed_text"].strip()])
    minutes = total_seconds / 60
    print(f"Dataset: {csv_path} ({minutes:.1f} min of audio)")
    if minutes < 10:
        print(
            f"WARNING: {minutes:.1f} min is below the recommended 10-20 min minimum "
            "(see module docstring - this is an open/unverified threshold, not a hard rule, "
            "but don't expect good quality from a very short recording).",
            file=sys.stderr,
        )
    return csv_path


def run_training(
    csv_path: Path,
    audio_dir: Path,
    cache_dir: Path,
    output_dir: Path,
    voice_name: str,
    espeak_voice: str,
    sample_rate: int,
    max_steps: int,
    ckpt_path: str | None,
    train_python: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    # PyTorch 2.6+ defaults torch.load(weights_only=True), which rejects the pathlib.PosixPath
    # object embedded in rhasspy/piper-checkpoints .ckpt files. Fix via the officially-recommended
    # safe_globals allowlist (not the riskier weights_only=False) by shimming the CLI entry point.
    # Also: the packaged val_mos ModelCheckpoint callback hard-crashes
    # (MisconfigurationException) instead of soft-skipping when the MOS predictor
    # doesn't log in time (observed even after downloading SpeechMOS and adding
    # num_test_examples). Not essential -- val_mel + save_last still select checkpoints.
    # Strip it from the hardcoded callback list before main() builds the CLI/Trainer.
    shim = (
        "import pathlib, torch.serialization, sys; "
        "torch.serialization.add_safe_globals([pathlib.PosixPath]); "
        "import piper.train.__main__ as m; "
        "m._DEFAULT_CALLBACKS = [c for c in m._DEFAULT_CALLBACKS "
        "if getattr(c, 'monitor', None) != 'val_mos']; "
        "from piper.train.__main__ import main; sys.argv = ['piper.train'] + sys.argv[1:]; main()"
    )
    cmd = [
        train_python, "-c", shim, "fit",
        "--data.voice_name", voice_name,
        "--data.csv_path", str(csv_path),
        "--data.audio_dir", str(audio_dir),
        "--model.sample_rate", str(sample_rate),
        "--data.espeak_voice", espeak_voice,
        "--data.cache_dir", str(cache_dir),
        "--data.config_path", str(output_dir / "config.json"),
        "--data.batch_size", "8",
        # num_test_examples=0 crashes: the val_mos ModelCheckpoint callback hard-fails
        # (MisconfigurationException) instead of the docstring's assumed soft-skip when
        # val_mos is never logged at all. >=1 test example is needed for it to log something.
        "--data.num_test_examples", "2",
        "--trainer.max_steps", str(max_steps),
        "--trainer.accelerator", "cpu",  # verified faster than mps on this hardware, see docstring
        "--trainer.enable_checkpointing", "true",
        "--trainer.default_root_dir", str(output_dir),
        "--trainer.log_every_n_steps", "10",
    ]
    if ckpt_path:
        # NOT --ckpt_path: that's for resuming an IDENTICAL run and eagerly re-parses the
        # checkpoint's saved hyperparameters, which fails schema-mismatch ("sample_bytes")
        # against a pretrained checkpoint from a different piper1-gpl version. warmstart_ckpt
        # is the documented mechanism for initializing weights from a different checkpoint.
        cmd += ["--model.warmstart_ckpt", ckpt_path]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def export_onnx(checkpoint: Path, output_onnx: Path, train_python: str):
    # Piper's packaged `piper.train.export_onnx` uses torch's new default ONNX exporter,
    # which fails on this model (see module docstring). Force the legacy exporter instead.
    export_script = f"""
import pathlib
import torch
import torch.serialization
torch.serialization.add_safe_globals([pathlib.PosixPath])
from piper.train.vits.lightning import VitsModel

model = VitsModel.load_from_checkpoint(r"{checkpoint}", map_location="cpu")
model_g = model.model_g
model_g.eval()
with torch.no_grad():
    model_g.dec.remove_weight_norm()

def infer_forward(text, text_lengths, scales, sid=None):
    audio = model_g.infer(
        text, text_lengths,
        noise_scale=scales[0], length_scale=scales[1], noise_scale_w=scales[2], sid=sid,
    )[0].unsqueeze(1)
    return audio

model_g.forward = infer_forward
num_symbols = model_g.n_vocab
num_speakers = model_g.n_speakers
dummy_input_length = 50
sequences = torch.randint(low=0, high=num_symbols, size=(1, dummy_input_length), dtype=torch.long)
sequence_lengths = torch.LongTensor([sequences.size(1)])
sid = torch.LongTensor([0]) if num_speakers > 1 else None
scales = torch.FloatTensor([0.667, 1.0, 0.8])

torch.onnx.export(
    model=model_g, args=(sequences, sequence_lengths, scales, sid), f=r"{output_onnx}",
    verbose=False, opset_version=15,
    input_names=["input", "input_lengths", "scales", "sid"], output_names=["output"],
    dynamic_axes={{
        "input": {{0: "batch_size", 1: "phonemes"}},
        "input_lengths": {{0: "batch_size"}},
        "output": {{0: "batch_size", 2: "time"}},
    }},
    dynamo=False,
)
print("Exported", r"{output_onnx}")
"""
    subprocess.run([train_python, "-c", export_script], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--language", required=True, help="e.g. en, sk - matches speaker_voices.json 'language' field")
    p.add_argument("--voice-name", required=True, help="output voice id, e.g. en_US-personal-medium")
    p.add_argument("--espeak-voice", required=True, help="espeak-ng voice, e.g. en-us, sk")
    p.add_argument("--speaker-voices-dir", default="speaker_voices")
    p.add_argument("--work-dir", default="/tmp/piper_finetune_work")
    p.add_argument("--output-dir", default="backend/tts/piper_models")
    p.add_argument("--sample-rate", type=int, default=22050)
    p.add_argument("--max-steps", type=int, default=2000, help="ballpark, see module docstring - re-tune after first real run")
    p.add_argument("--ckpt-path", default=None, help="pretrained Piper checkpoint to fine-tune from (strongly recommended, see TRAINING.md)")
    p.add_argument("--train-python", default=sys.executable, help="python from the SEPARATE training venv, not the main project venv")
    args = p.parse_args()

    speaker_voices_dir = Path(args.speaker_voices_dir)
    work_dir = Path(args.work_dir)
    output_dir = Path(args.output_dir)

    entries = read_speaker_voices(speaker_voices_dir, args.language)
    if not entries:
        sys.exit(f"No transcribed {args.language} recordings found in {speaker_voices_dir}/speaker_voices.json")

    csv_path = build_dataset_csv(entries, speaker_voices_dir, work_dir / "data")
    run_training(
        csv_path=csv_path,
        audio_dir=speaker_voices_dir,
        cache_dir=work_dir / "cache",
        output_dir=work_dir / "out",
        voice_name=args.voice_name,
        espeak_voice=args.espeak_voice,
        sample_rate=args.sample_rate,
        max_steps=args.max_steps,
        ckpt_path=args.ckpt_path,
        train_python=args.train_python,
    )

    # BUG FIXED 2026-08-19: was hardcoded to "version_0", but Lightning auto-increments the
    # version dir on every run sharing the same work_dir (retries after a crash included) --
    # silently exporting the wrong (earlier, less-trained) checkpoint otherwise. Pick the
    # highest version_N with a last.ckpt actually present.
    lightning_logs = work_dir / "out" / "lightning_logs"
    version_dirs = sorted(
        (d for d in lightning_logs.glob("version_*") if (d / "checkpoints" / "last.ckpt").exists()),
        key=lambda d: int(d.name.split("_")[1]),
    )
    if not version_dirs:
        sys.exit(f"No trained checkpoint found under {lightning_logs}")
    ckpt = version_dirs[-1] / "checkpoints" / "last.ckpt"
    print(f"Using {ckpt} (latest of {len(version_dirs)} version dir(s) found)")
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_out = output_dir / f"{args.voice_name}.onnx"
    export_onnx(ckpt, onnx_out, args.train_python)
    (output_dir / f"{args.voice_name}.onnx.json").write_text((work_dir / "out" / "config.json").read_text())
    print(f"Done. Personal voice ready at {onnx_out} - selectable like any other Piper voice.")


if __name__ == "__main__":
    main()
