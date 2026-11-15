#!/usr/bin/env python3
"""Standalone X-Voice T4 benchmark. No BP imports or project-environment installs.

Source: https://github.com/sunnyxrxrx/X-Voice/tree/817bcdd9f8173eb01c7d5c895858b1d93780d9d0
README.md; infer_cli_stage{1,2}.py; utils_infer.py; XVoice_Base_Stage{1,2}.yaml.
Code MIT; HF XRXRX/X-Voice weights CC-BY-NC. Batch only, no streaming.
Run only on Kaggle; this file re-executes itself inside isolated Python 3.11 venvs.
"""
import json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

OUT = Path("/kaggle/working")
REPORT = OUT / "results.json"
T0 = time.time()
R = {"schema": 1, "status": "running", "steps": [], "errors": [], "tts": {}, "infrastructure": {}}
OUT.mkdir(parents=True, exist_ok=True)


def save():
    R["elapsed_s"] = round(time.time() - T0, 1)
    tmp = REPORT.with_suffix(".tmp")
    tmp.write_text(json.dumps(R, indent=1, ensure_ascii=False))
    tmp.replace(REPORT)


def step(name, **kw):
    R["steps"].append({"t": round(time.time() - T0, 1), "step": name, **kw})
    print(f"[{time.time() - T0:7.1f}s] {name} {kw}", flush=True)
    save()

import gc
import statistics
import unicodedata
import wave

REV = "817bcdd9f8173eb01c7d5c895858b1d93780d9d0"
SOURCE = Path("/tmp/xvoice-source")
TTS_PY = "/tmp/xvoice-venv/bin/python"
ASR_PY = "/tmp/xvoice-asr-venv/bin/python"
SELF = str(Path(__file__).resolve())
LANGS = ("en", "sk", "cs")
TEXTS = {
    "en": {
        "short": "Hello, this is a test of real-time speech translation.",
        "long": "During our international meeting today, each participant will hear translated speech in their own language while the system preserves the original speaker's voice and meaning.",
    },
    "sk": {
        "short": "Dobrý deň, toto je test prekladov reči v reálnom čase.",
        "long": "Počas dnešného medzinárodného stretnutia bude každý účastník počuť preloženú reč vo svojom jazyku, pričom systém zachová hlas pôvodného rečníka a presný význam všetkých jeho slov.",
    },
    "cs": {
        "short": "Dobrý den, toto je test překladu řeči v reálném čase.",
        "long": "Během dnešního mezinárodního setkání bude každý účastník slyšet přeloženou řeč ve svém jazyce, přičemž systém zachová hlas původního řečníka a přesný význam všech jeho slov.",
    },
}
CANDIDATES = {
    "en": ["speaker_voices/me_en_rainbow.m4a", "processed/testaudio/en_paced.wav"],
    "sk": ["speaker_voices/me_sk_trhove.m4a", "processed/testaudio/sk_paced.wav"],
    "cs": ["speaker_voices/me_cs_staromestske.m4a", "processed/testaudio/cs_paced.wav"],
}


def error(where, exc, **context):
    record = {"stage": where, "error": f"{type(exc).__name__}: {exc}", **context}
    R["errors"].append(record)
    step("error", **record)
    return record


def run(cmd, timeout=900, cwd=None, env=None):
    """No shell interpolation; record commands, return codes and complete disk logs."""
    name = f"command_{len(R['steps']):04d}.log"
    log = OUT / name
    started = time.perf_counter()
    with log.open("w") as stream:
        try:
            proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=stream,
                                  stderr=subprocess.STDOUT, timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = 124
    step("command", command=cmd, rc=rc, wall_s=time.perf_counter() - started,
         log=str(log), tail=log.read_text(errors="replace")[-2000:])
    if rc:
        raise RuntimeError(f"command rc={rc}: {cmd}; see {log}")
    return log.read_text(errors="replace")


def isolated_env():
    env = os.environ.copy()
    for name in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    env.update(HF_HOME="/tmp/xvoice-hf", TORCH_HOME="/tmp/xvoice-torch",
               XDG_CACHE_HOME="/tmp/xvoice-cache", UV_CACHE_DIR="/tmp/xvoice-uv",
               PYTHONNOUSERSITE="1", TOKENIZERS_PARALLELISM="false")
    return env


def prepare_references():
    references = {}
    for lang in LANGS:
        try:
            roots = [Path.cwd()]
            # Reuse old staged-repo discovery; also handle datasets without app.py.
            roots += sorted({p.parent for p in Path("/kaggle/input").rglob("app.py")
                             if (p.parent / "benchmark-provenance.json").is_file()})
            matches = [root / rel for root in roots for rel in CANDIDATES[lang]
                       if (root / rel).is_file()]
            if not matches:
                matches = sorted(Path("/kaggle/input").rglob(f"me_{lang}_*.m4a"))
            if not matches:
                matches = sorted(Path("/kaggle/input").rglob(f"{lang}_paced.wav"))
            if not matches:
                raise FileNotFoundError(f"No real {lang} reference in {CANDIDATES[lang]}")
            dest = OUT / f"ref_{lang}.wav"
            run(["ffmpeg", "-y", "-v", "error", "-i", str(matches[0]), "-t", "10",
                 "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(dest)], 60)
            with wave.open(str(dest)) as wav:
                duration = wav.getnframes() / wav.getframerate()
                if wav.getnchannels() != 1 or abs(duration - 10) > 0.01:
                    raise ValueError(f"Reference must be 10 s mono, got {duration} s")
            references[lang] = {"source": str(matches[0]), "wav": str(dest),
                                "audio_s": duration}
            step("reference", lang=lang, **references[lang])
        except Exception as exc:
            error("reference", exc, lang=lang)
    R["references"] = references
    save()
    if not references:
        raise RuntimeError("No valid real reference clips; no synthetic fallback allowed")


def asr_worker(mode):
    """ASR uses CPU int8 in its own venv, never X-Voice's CT2/torch environment."""
    from faster_whisper import WhisperModel
    started = time.perf_counter()
    model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8",
                         download_root="/tmp/xvoice-asr-models")
    R["asr"] = {"model": "large-v3-turbo", "device": "cpu", "compute_type": "int8",
                "model_load_s": time.perf_counter() - started, "items": {}}
    manifest = json.loads((OUT / "manifest.json").read_text())
    items = manifest["references"] if mode == "references" else manifest["outputs"]
    for key, item in items.items():
        try:
            started = time.perf_counter()
            segments, info = model.transcribe(item["wav"], language=item["lang"],
                                               beam_size=5, vad_filter=False)
            text = " ".join(segment.text.strip() for segment in segments).strip()
            if not text:
                raise ValueError("Empty ASR transcript")
            result = {"text": text, "transcription_s": time.perf_counter() - started,
                      "language": info.language}
            if mode == "outputs":
                expected = normalise(item["text"])
                actual = normalise(text)
                result.update(reference_words=expected, hypothesis_words=actual,
                              wer=word_error(expected, actual))
            R["asr"]["items"][key] = result
            step("asr_done", item=key, **result)
        except Exception as exc:
            R["asr"]["items"][key] = error("asr", exc, item=key)
    save()


def normalise(text):
    # Preserve diacritics, lowercase and strip Unicode punctuation before splitting.
    return "".join(c for c in text.lower()
                   if not unicodedata.category(c).startswith("P")).split()


def word_error(reference, hypothesis):
    row = list(range(len(hypothesis) + 1))
    for i, word in enumerate(reference, 1):
        new = [i]
        for j, other in enumerate(hypothesis, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (word != other)))
        row = new
    return row[-1] / len(reference)


def tts_worker():
    # Import once; both stages, all languages and repeats share this one process.
    import numpy as np
    import soundfile as sf
    import torch
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    from x_voice.infer import utils_infer as u

    manifest = json.loads((OUT / "manifest.json").read_text())
    R.update(models={}, outputs={}, auto_detect_lang=False,
             timing="non-streaming; first_audio_latency_s equals full inference wall_s",
             reference_preprocessing="10 s mono crop; upstream silence trimming disabled")
    if not torch.cuda.is_available() or "sm_75" not in torch.cuda.get_arch_list():
        raise RuntimeError("CUDA T4 sm_75 support required")
    device = "cuda:0"
    R["infrastructure"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                            "arch_list": torch.cuda.get_arch_list(),
                            "gpu": torch.cuda.get_device_name(0)}
    for stage in ("stage1", "stage2"):
        model = vocoder = srp = getter = None
        try:
            cfg = OmegaConf.load(SOURCE / f"src/x_voice/configs/XVoice_Base_{stage.title()}.yaml")
            arc = OmegaConf.to_container(cfg.model.arch, resolve=True)
            model_cls = get_class(f"x_voice.model.{cfg.model.backbone}")
            tokenizer = cfg.model.tokenizer
            kwargs = dict(mel_spec_type=cfg.model.mel_spec.mel_spec_type,
                          vocab_file=manifest["weights"]["vocab"], device=device,
                          tokenizer=tokenizer, tokenizer_path=cfg.model.get("tokenizer_path"),
                          dataset_name=cfg.datasets.name,
                          mel_spec_kwargs=OmegaConf.to_container(cfg.model.mel_spec, resolve=True))
            torch.cuda.synchronize()
            started = time.perf_counter()
            # Public CLI's model construction, with explicit HF checkpoint paths.
            if stage == "stage1":
                model = u.load_model(model_cls, arc, manifest["weights"][stage], **kwargs)
            else:
                if not cfg.model.sft or cfg.model.use_total_text:
                    raise ValueError("Expected Stage2 drop-text config")
                model = u.load_model_sft(model_cls, arc, manifest["weights"][stage],
                                         use_total_text=False, **kwargs)
            model.eval()
            torch.cuda.synchronize()
            model_s = time.perf_counter() - started
            started = time.perf_counter()
            # YAML's local vocoder path is absent on Kaggle; use documented HF loader.
            vocoder = u.load_vocoder(vocoder_name="vocos", is_local=False,
                                     device=device, hf_cache_dir="/tmp/xvoice-hf")
            vocoder.eval()
            torch.cuda.synchronize()
            vocoder_s = time.perf_counter() - started
            srp_s = 0.0
            if stage == "stage2":
                started = time.perf_counter()
                srp = u.load_srp_model(
                    str(SOURCE / "src/rate_pred/configs/SpeedPredict_Multilingual.yaml"),
                    manifest["weights"]["srp"], device)
                srp.eval()
                torch.cuda.synchronize()
                srp_s = time.perf_counter() - started
            load_s = model_s + vocoder_s + srp_s
            R["models"][stage] = dict(model_load_s=load_s, load_model_s=model_s,
                                       load_vocoder_s=vocoder_s, load_srp_s=srp_s)
            getter = u.get_ipa_tokenizer_cache(tokenizer, cfg.model.get("stress", True))
            lang_map = getattr(model.transformer, "lang_to_id", {})
            step("model_loaded", model=stage, **R["models"][stage])
            for lang in LANGS:
                for nfe in (32, 16):
                    for length, text in TEXTS[lang].items():
                        key = f"xvoice_{stage}_nfe{nfe}_{lang}_{length}"
                        entry = {"lang": lang, "text": text, "nfe_step": nfe,
                                 "model": stage, "model_load_s": load_s,
                                 "streaming": False, "status": "running", "repeats": []}
                        R["tts"][key] = entry
                        try:
                            if lang not in cfg.model.arch.languages:
                                entry["status"] = "unsupported"
                                continue
                            ref = manifest["references"][lang]
                            if not ref.get("text", "").strip():
                                raise ValueError("Reference ASR transcript required; auto-ASR forbidden")
                            # Stage2 CLI parses [main|lang] into text + nested language spans.
                            tagged = f"[main|{lang}]{text}"
                            voice, parsed_lang = u.parse_voice_lang_tag(
                                tagged[1:tagged.index("]")], voice_names=["main"])
                            if voice != "main" or parsed_lang != lang:
                                raise ValueError("Unexpected Stage2 language-tag parse")
                            gen_text = tagged.split("]", 1)[1] if stage == "stage2" else text
                            common = dict(nfe_step_value=nfe, device_name=device,
                                          # Also bypass internal fastlid calls despite explicit spans.
                                          dominant_lang=[lang], layered=(stage == "stage2"),
                                          cfg_strength_value=2.0 if stage == "stage1" else 2.5,
                                          cfg_strength2_value=0.0 if stage == "stage1" else 4.0,
                                          cfg_schedule_value=None if stage == "stage1" else "square",
                                          cfg_decay_time_value=0.6)
                            def synthesize():
                                args = ([gen_text], [[(lang, gen_text)]], tokenizer, getter,
                                        model, vocoder, lang_map)
                                if stage == "stage1":
                                    return u.infer_xvoice_process(
                                        ref["wav"], ref["text"], args[0], lang, *args[1:],
                                        sp_type="syllable", **common)
                                return u.infer_xvoice_droptext_process(
                                    ref["wav"], *args, srp, **common)

                            # One warm-up per stage/language/NFE/text, excluded from medians.
                            with torch.inference_mode():
                                torch.manual_seed(1234)
                                warm = synthesize()
                                torch.cuda.synchronize()
                                del warm
                            entry["warmup_complete"] = True
                            for repeat in range(3):
                                try:
                                    torch.manual_seed(1234 + repeat)
                                    torch.cuda.synchronize()
                                    torch.cuda.reset_peak_memory_stats()
                                    started = time.perf_counter()
                                    with torch.inference_mode():
                                        audio, sr, _ = synthesize()
                                    torch.cuda.synchronize()
                                    wall = time.perf_counter() - started
                                    peak = torch.cuda.max_memory_allocated() / 1024**2
                                    # Upstream returns list of arrays for batched text input.
                                    if isinstance(audio, list):
                                        audio = np.concatenate(audio)
                                    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
                                    duration = len(audio) / sr
                                    if duration <= 0 or not np.isfinite(audio).all():
                                        raise ValueError("Invalid or empty generated audio")
                                    wav = OUT / f"{key}_r{repeat + 1}.wav"
                                    sf.write(wav, audio, sr)
                                    item = dict(wall_s=wall, first_audio_latency_s=wall,
                                                audio_s=duration, rtf=wall / duration,
                                                peak_gpu_mb=peak, wav=str(wav), repeat=repeat + 1)
                                    entry["repeats"].append(item)
                                    R["outputs"][wav.stem] = dict(wav=str(wav), lang=lang, text=text)
                                    save()
                                    del audio
                                except Exception as exc:
                                    error("repeat", exc, key=key, repeat=repeat + 1)
                            if len(entry["repeats"]) != 3:
                                raise RuntimeError("Incomplete repeats; no three-repeat median")
                            for metric in ("wall_s", "first_audio_latency_s", "audio_s", "rtf", "peak_gpu_mb"):
                                entry[metric] = statistics.median(x[metric] for x in entry["repeats"])
                            representative = sorted(entry["repeats"], key=lambda x: x["wall_s"])[1]
                            dest = OUT / f"{key}.wav"
                            shutil.copyfile(representative["wav"], dest)
                            entry.update(output_wav=str(dest), status="ok",
                                         representative_repeat=representative["repeat"])
                            R["outputs"][key] = dict(wav=str(dest), lang=lang, text=text)
                            if stage == "stage1" and nfe == 32 and length == "short":
                                alias = OUT / f"xvoice_{lang}.wav"
                                shutil.copyfile(dest, alias)
                                entry["default_output_wav"] = str(alias)
                            step("tts_done", key=key, wall_s=entry["wall_s"], rtf=entry["rtf"])
                        except Exception as exc:
                            entry.update(status="failed", error=str(exc))
                            error("tts", exc, key=key)
                        finally:
                            save()
        except Exception as exc:
            R["models"][stage] = error("model_load", exc, model=stage)
            for lang in LANGS:
                R["tts"][f"{stage}_{lang}"] = {"status": "failed", "error": str(exc)}
        finally:
            model = vocoder = srp = getter = None
            gc.collect()
            torch.cuda.empty_cache()
            save()


def install_dependencies():
    """Read the real upstream dependency list; bypass only its CT2 pin."""
    import tomllib
    with (SOURCE / "pyproject.toml").open("rb") as stream:
        deps = tomllib.load(stream)["project"]["dependencies"]
    constraints = OUT / "torch-constraints.txt"
    probe = json.loads(subprocess.check_output(
        [TTS_PY, "-c", "import torch; print(__import__('json').dumps({'version':torch.__version__}))"], text=True))
    version = probe["version"].split("+")[0]
    codec = "0.7.0" if version == "2.8.0" else "0.5.0"
    constraints.write_text(f"torch=={version}\ntorchaudio=={version}\ntorchcodec=={codec}\n")
    deps = ["ctranslate2" if d == "ctranslate2==4.5.0" else d for d in deps]
    run(["uv", "pip", "install", "--python", TTS_PY, "--constraint", str(constraints),
         *deps], 1800, env=isolated_env())
    run(["uv", "pip", "install", "--python", TTS_PY, "--no-deps", str(SOURCE)],
        300, env=isolated_env())
    # 2026-11-15 run: newest pyarrow demanded NumPy>=2 while X-Voice's pins left 1.26.4; keep a pyarrow that works with numpy 1.x
    run(["uv", "pip", "install", "--python", TTS_PY, "pyarrow<20", "six"], 300, env=isolated_env())
    R["infrastructure"]["dependency_strategy"] = "upstream deps with unpinned CT2; repo --no-deps"
    run(["uv", "pip", "freeze", "--python", TTS_PY], env=isolated_env())


def download_weights():
    from huggingface_hub import hf_hub_download
    names = {"stage1": "XVoice_Base_Stage1/model_600000.safetensors",
             "stage2": "XVoice_Base_Stage2/model_70000.safetensors",
             "vocab": "XVoice_Base_Stage1/vocab.txt",
             "srp": "SpeedPredictor/model_28000.safetensors"}
    weights = {}
    for key, name in names.items():
        try:
            weights[key] = hf_hub_download("XRXRX/X-Voice", name,
                                           cache_dir="/tmp/xvoice-hf")
            step("weight", name=name, path=weights[key])
        except Exception as exc:
            error("weight", exc, name=name)
    R["weights"] = weights
    save()


def child(mode, python, report):
    try:
        run([python, SELF, mode, str(report)], 3600, env=isolated_env())
    except Exception:
        if report.is_file():
            R.setdefault("failed_children", {})[mode] = json.loads(report.read_text())
            save()
        raise
    return json.loads(report.read_text())


def main():
    R.update(schema=2, source_revision=REV, licence={"code": "MIT", "weights": "CC-BY-NC"},
             texts=TEXTS, repeats=3, warmups=1)
    try:
        step("env", python=sys.version, cwd=os.getcwd())
        run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], 60)
        prepare_references()
        run([sys.executable, "-m", "pip", "install", "uv"], 180)
        run(["uv", "python", "install", "3.11"], 300, env=isolated_env())
        for path in ("/tmp/xvoice-venv", "/tmp/xvoice-asr-venv"):
            run(["uv", "venv", "--python", "3.11", path], 120, env=isolated_env())
        run(["git", "clone", "https://github.com/sunnyxrxrx/X-Voice.git", str(SOURCE)], 300)
        run(["git", "checkout", "--detach", REV], 60, cwd=str(SOURCE))
        for version, index in (("2.8.0", "cu128"), ("2.7.1", "cu128"), ("2.7.1", "cu126")):
            try:
                run(["uv", "pip", "install", "--python", TTS_PY, "--reinstall",
                     f"torch=={version}", f"torchaudio=={version}", "--index-url",
                     f"https://download.pytorch.org/whl/{index}"], 900, env=isolated_env())
                run([TTS_PY, "-c", "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list()); assert torch.cuda.is_available(); assert 'sm_75' in torch.cuda.get_arch_list(); x=torch.ones(1,device='cuda'); print((x+x).cpu())"], 60,
                    env=isolated_env())
                R["infrastructure"]["torch_build"] = {"version": version, "index": index}
                break
            except Exception as exc:
                error("torch_candidate", exc, version=version, index=index)
        else:
            raise RuntimeError("No tested torch build supports T4 sm_75")
        child("deps", TTS_PY, OUT / "deps.json")
        R["infrastructure"]["dependency_strategy"] = "unpinned CT2, constrained torch, repo --no-deps"
        env = isolated_env()
        env["PATH"] = f"/tmp/xvoice-venv/bin:{env['PATH']}"
        try:
            run(["bash", "src/x_voice/prepare_ipa.sh"], 1200, cwd=str(SOURCE), env=env)
        except RuntimeError as exc:
            # the script's final self-test imports x_voice; espeak-ng itself may be installed fine (it was on 2026-11-15)
            R["infrastructure"]["prepare_ipa_error"] = str(exc)[:300]
            run(["espeak-ng", "--version"], 30, env=env)
        # ASR installs are completely separate from the X-Voice resolver.
        run(["uv", "pip", "install", "--python", ASR_PY, "faster-whisper"], 600, env=isolated_env())
        run(["uv", "pip", "freeze", "--python", ASR_PY], env=isolated_env())
        manifest = {"references": {lang: dict(ref, lang=lang)
                                    for lang, ref in R["references"].items()}}
        manifest_path = OUT / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False))
        refs = child("references", ASR_PY, OUT / "reference_asr.json")
        R["reference_asr"] = refs
        for lang, ref in manifest["references"].items():
            item = refs["asr"]["items"].get(lang, {})
            ref.update(text=item.get("text", ""), transcription_s=item.get("transcription_s"))
        weights = child("weights", TTS_PY, OUT / "weights.json")
        manifest["weights"] = weights["weights"]
        R["weights"] = weights
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False))
        tts = child("tts", TTS_PY, OUT / "tts.json")
        R["tts"] = tts["tts"]
        R["models"] = tts.get("models", {})
        R["tts_infrastructure"] = tts["infrastructure"]
        R["errors"].extend(tts["errors"])
        manifest["outputs"] = tts.get("outputs", {})
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False))
        try:
            asr = child("outputs", ASR_PY, OUT / "output_asr.json")
            R["round_trip_asr"] = asr
            R["errors"].extend(asr["errors"])
            for key, entry in R["tts"].items():
                if entry.get("status") != "ok":
                    continue
                items = asr["asr"]["items"]
                entry["asr"] = items.get(key)
                wers = []
                for repeat in entry["repeats"]:
                    repeat["asr"] = items.get(Path(repeat["wav"]).stem)
                    if repeat["asr"] and "wer" in repeat["asr"]:
                        wers.append(repeat["asr"]["wer"])
                entry["wer_median"] = statistics.median(wers) if len(wers) == 3 else None
                entry["asr_status"] = "ok" if len(wers) == 3 else "incomplete"
        except Exception as exc:
            error("round_trip_asr", exc)
        good = sum(entry.get("status") == "ok" and entry.get("asr_status") == "ok"
                   for entry in R["tts"].values())
        R["successful_configurations"] = good
        R["expected_configurations"] = 24
        R["status"] = "complete" if good == 24 else "partial" if good else "failed"
    except Exception as exc:
        error("setup", exc)
        for lang in LANGS:
            R["tts"].setdefault(f"setup_{lang}", {"status": "blocked", "error": str(exc)})
        R["status"] = "failed"
    finally:
        save()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        REPORT = Path(sys.argv[2])
        try:
            {"deps": install_dependencies, "weights": download_weights,
             "tts": tts_worker, "references": lambda: asr_worker("references"),
             "outputs": lambda: asr_worker("outputs")}[sys.argv[1]]()
            R["status"] = "partial" if R["errors"] else "complete"
        except Exception as exc:
            error(sys.argv[1], exc)
            R["status"] = "failed"
            raise
        finally:
            save()
    else:
        main()
