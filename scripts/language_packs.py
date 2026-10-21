#!/usr/bin/env python3
"""Language packs — the declarative unit of a supported language pair.

A pair is supported when its pack resolves (STT + MT + TTS all present) and it
has a passing eval record under processed/language_eval/. Adding a language is
adding a JSON file in languages/packs/, not editing the pipeline.

    python scripts/language_packs.py list
    python scripts/language_packs.py show en-sk
    python scripts/language_packs.py add pl-en
    python scripts/language_packs.py remove pl-en --yes
    python scripts/language_packs.py test en-sk [--clip PATH] [--ref "ground truth text"]

`test` runs STT -> MT -> TTS on the pack's clip and writes
processed/language_eval/<pair>.json with per-stage latency, the transcript, the
translation, and WER when a reference is given. It never fakes a pass: if a
component is missing or the clip is absent, it records status=experimental with
the reason.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKS_DIR = os.path.join(REPO_ROOT, "languages", "packs")
CT2_DIR = os.path.join(REPO_ROOT, "ct2_models")
PIPER_DIR = os.path.join(REPO_ROOT, "backend", "tts", "piper_models")
EVAL_DIR = os.path.join(REPO_ROOT, "processed", "language_eval")

# faster-whisper fetches these from HuggingFace automatically; anything else must
# exist locally under ct2_models/whisper-<name>/. Discovered at runtime so the list
# does not go stale when faster-whisper adds sizes (it did: large-v3-turbo).
_FALLBACK_HF_WHISPER = {
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large-v1", "large-v2", "large-v3", "large",
    "large-v3-turbo", "distil-small.en", "distil-medium.en", "distil-large-v3",
}
_HF_WHISPER_CACHE: set | None = None


def hf_whisper_models() -> set:
    global _HF_WHISPER_CACHE
    if _HF_WHISPER_CACHE is None:
        try:
            from faster_whisper.utils import available_models
            _HF_WHISPER_CACHE = set(available_models())
        except Exception:
            _HF_WHISPER_CACHE = set(_FALLBACK_HF_WHISPER)
    return _HF_WHISPER_CACHE


# ---------------------------------------------------------------- pack I/O

def pack_path(pair: str) -> str:
    return os.path.join(PACKS_DIR, f"{pair}.json")


def load(pair: str) -> dict:
    """Raises FileNotFoundError (never sys.exit) so the module stays importable/testable."""
    p = pack_path(pair)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"no pack for '{pair}' ({p}) — try: bp language add {pair}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def all_packs() -> list[dict]:
    out = []
    if os.path.isdir(PACKS_DIR):
        for name in sorted(os.listdir(PACKS_DIR)):
            if name.endswith(".json"):
                with open(os.path.join(PACKS_DIR, name), encoding="utf-8") as f:
                    out.append(json.load(f))
    return out


# ---------------------------------------------------------------- resolve

def resolve_stt(cfg: dict) -> tuple[str | None, str | None]:
    model = cfg.get("model", "")
    local = os.path.join(CT2_DIR, f"whisper-{model}")
    if os.path.exists(os.path.join(local, "model.bin")):
        return model, None
    if model in hf_whisper_models():
        return model, None  # faster-whisper downloads it on first use
    return None, f"STT model '{model}' not local and not a HF whisper size"


def resolve_mt(cfg: dict) -> tuple[str | None, str | None]:
    model = cfg.get("model", "")
    if not model:
        return None, "no MT model declared"
    if "/" in model:
        return None, f"MT '{model}' needs conversion (run scripts/convert_models.py)"
    path = os.path.join(CT2_DIR, model)
    if not os.path.exists(os.path.join(path, "model.bin")):
        return None, f"MT '{model}' converted dir not found in ct2_models/"
    # A dir with model.bin but no tokenizer files is a broken/partial conversion
    # (real case: Helsinki-NLP--opus-mt-en-cs) — it would fail at load time.
    has_tokenizer = any(os.path.exists(os.path.join(path, f))
                        for f in ("vocab.json", "source.spm"))
    if not has_tokenizer:
        return None, f"MT '{model}' is missing tokenizer files (re-run scripts/convert_models.py)"
    return model, None


def resolve_tts(cfg: dict) -> tuple[str | None, str | None]:
    for voice in (cfg.get("voice"), cfg.get("fallback")):
        if voice and os.path.exists(os.path.join(PIPER_DIR, f"{voice}.onnx")):
            return voice, None
    return None, f"piper voice not found: {cfg.get('voice')}"


def resolve(pack: dict) -> dict:
    stt, stt_err = resolve_stt(pack.get("stt", {}))
    mt, mt_err = resolve_mt(pack.get("mt", {}))
    tts, tts_err = resolve_tts(pack.get("tts", {}))
    errors = [e for e in (stt_err, mt_err, tts_err) if e]
    return {
        "stt": stt, "mt": mt, "tts": tts,
        "errors": errors,
        "resolved": not errors,
    }


def eval_path(pair: str) -> str:
    return os.path.join(EVAL_DIR, f"{pair}.json")


def status(pack: dict) -> str:
    """verified | experimental — verified requires a recorded passing eval."""
    r = resolve(pack)
    if not r["resolved"]:
        return "experimental"
    ep = eval_path(pack["pair"])
    if os.path.isfile(ep):
        try:
            with open(ep, encoding="utf-8") as f:
                rec = json.load(f)
            if rec.get("status") == "verified":
                return "verified"
        except (OSError, json.JSONDecodeError):
            pass
    return "experimental"


# ---------------------------------------------------------------- commands

def cmd_list(args: argparse.Namespace) -> None:
    packs = all_packs()
    if not packs:
        print("no packs in languages/packs/")
        return
    print(f"{'pair':8s} {'stt':>10s} {'mt':>28s} {'tts':>30s}  status")
    for pack in packs:
        r = resolve(pack)
        def mark(v):
            return v if v else "--"
        print(f"{pack['pair']:8s} {mark(r['stt']):>10s} {mark(r['mt']):>28s} "
              f"{mark(r['tts']):>30s}  {status(pack)}")
    print()
    print("verified = components resolve AND a passing eval record exists;")
    print("experimental = missing component or no recorded eval. Detail: bp language show <pair>")


def cmd_show(args: argparse.Namespace) -> None:
    pack = load(args.pair)
    print(json.dumps(pack, indent=2, ensure_ascii=False))
    r = resolve(pack)
    print("\n# resolved:", json.dumps(r, ensure_ascii=False))
    print("# status:", status(pack))
    ep = eval_path(args.pair)
    if os.path.isfile(ep):
        print(f"# last eval: {ep}")


def _detect_tts(target: str) -> str | None:
    if not os.path.isdir(PIPER_DIR):
        return None
    names = [n[:-5] for n in sorted(os.listdir(PIPER_DIR))
             if n.endswith(".onnx") and n.startswith(f"{target}_")]
    if not names:
        return None
    # prefer the owner's personal / omni voice over a generic one of the same language
    for pref in ("personal", "omni"):
        for n in names:
            if pref in n:
                return n
    return names[0]


def hf_model_info(repo_id: str, timeout: float = 6.0) -> tuple[bool, str]:
    """(exists_on_hf, licence) for a HF model id. Best-effort: never blocks adding a
    pack offline (returns (True, 'unknown') when the network is unreachable)."""
    try:
        req = urllib.request.Request(f"https://huggingface.co/api/models/{repo_id}",
                                     headers={"User-Agent": "bp-language-add"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.load(r)
    except urllib.error.HTTPError as exc:
        return (False, "unknown") if exc.code == 404 else (True, "unknown")
    except Exception:
        return True, "unknown"
    lic = str((data.get("cardData") or {}).get("license") or "").lower()
    if "nc" in lic.replace("cc-by-", "cc-by") and ("non-commercial" in lic or "-nc" in lic):
        return True, "non-commercial"
    if lic:
        return True, "commercial-ok"
    return True, "unknown"


def cmd_add(args: argparse.Namespace) -> None:
    pair = args.pair
    if os.path.exists(pack_path(pair)):
        raise FileExistsError(f"pack already exists: {pack_path(pair)}")
    if "-" not in pair:
        raise ValueError("pair must look like 'en-sk' (source-target)")
    src, tgt = pair.split("-", 1)
    local_mt = f"Helsinki-NLP--opus-mt-{src}-{tgt}"
    warning = None
    if os.path.isdir(os.path.join(CT2_DIR, local_mt)):
        mt_model, license_ = local_mt, "commercial-ok"
    else:
        mt_model = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        exists, license_ = hf_model_info(mt_model)
        if not exists:
            warning = (f"MT model '{mt_model}' was NOT found on HuggingFace — the pair may not "
                       f"exist; fix the pack or convert a model, then re-run the test")
        elif license_ == "unknown":
            warning = (f"licence for '{mt_model}' could not be verified (HF returns HTTP 401 for "
                       f"unauthenticated lookups) — confirm the model card, or pass --license")
    if getattr(args, "license", None):
        license_ = args.license
    stt_model = "small-sk" if src == "sk" and \
        os.path.isdir(os.path.join(CT2_DIR, "whisper-small-sk")) else "base"
    pack = {
        "pair": pair, "source": src, "target": tgt,
        "stt": {"engine": "faster_whisper", "model": stt_model, "language": src},
        "mt": {"engine": "opus-mt", "model": mt_model},
        "tts": {"engine": "piper", "voice": _detect_tts(tgt) or f"{tgt}_XX-unknown-medium"},
        "clip": None, "reference": None, "license": license_,
        "notes": "scaffolded by bp language add — fill clip/reference, then run: bp language test " + pair,
    }
    if warning:
        pack["notes"] = warning + ". " + pack["notes"]
    os.makedirs(PACKS_DIR, exist_ok=True)
    with open(pack_path(pair), "w", encoding="utf-8") as f:
        json.dump(pack, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {pack_path(pair)}  (licence: {license_})")
    if warning:
        print(f"WARNING: {warning}")
    print(f"resolved: {json.dumps(resolve(pack), ensure_ascii=False)}")
    print(f"next: bp language test {pair}")


def cmd_remove(args: argparse.Namespace) -> None:
    p = pack_path(args.pair)
    if not os.path.exists(p):
        raise FileNotFoundError(f"no pack for {args.pair}")
    if not args.yes:
        raise PermissionError(f"refusing to delete {p} without --yes")
    os.remove(p)
    print(f"removed {p}")


def cmd_test(args: argparse.Namespace) -> None:
    pack = load(args.pair)
    r = resolve(pack)
    clip = args.clip or pack.get("clip")
    record = {
        "pair": pack["pair"], "source": pack["source"], "target": pack["target"],
        "engines": {"stt": r["stt"], "mt": r["mt"], "tts": r["tts"],
                    "stt_cfg": pack.get("stt"), "mt_cfg": pack.get("mt"),
                    "tts_cfg": pack.get("tts")},
        "clip": clip,
    }
    if not r["resolved"]:
        record.update({"status": "experimental", "reason": "; ".join(r["errors"])})
        _write_eval(pack["pair"], record)
        print(f"{pack['pair']}: EXPERIMENTAL — {'; '.join(r['errors'])}")
        return
    if not clip or not os.path.isfile(os.path.join(REPO_ROOT, clip)):
        record.update({"status": "experimental", "reason": f"clip not found: {clip}"})
        _write_eval(pack["pair"], record)
        print(f"{pack['pair']}: EXPERIMENTAL — clip not found: {clip}")
        return

    # heavy imports only on the real path
    import librosa
    import numpy as np
    from backend.stt.faster_whisper_stt import FasterWhisperSTT
    from backend.mt.ctranslate2_mt import CTranslate2MT
    from backend.tts.piper_tts import PiperTTS

    wav, sr = librosa.load(os.path.join(REPO_ROOT, clip), sr=16000, mono=True)
    stt = FasterWhisperSTT(model_size=pack["stt"]["model"])
    segs, t_stt, detected = stt.transcribe_audio(
        np.asarray(wav, dtype=np.float32), sr, language=pack["stt"]["language"])
    text = " ".join(s["text"] if isinstance(s, dict) else getattr(s, "text", "")
                    for s in segs).strip()

    mt_ref = pack["mt"]["model"]
    # CTranslate2MT takes a directory path (or an HF id with '/'), not a bare name.
    mt_path = mt_ref if "/" in mt_ref else os.path.join(CT2_DIR, mt_ref)
    mt = CTranslate2MT(model_path=mt_path)
    chunks = CTranslate2MT.split_chunks(text) or [text]
    translated_parts, _ = mt.translate_segments(chunks, pack["source"], pack["target"])
    translation = " ".join(translated_parts).strip()

    tts = PiperTTS(model_id=pack["tts"]["voice"])
    audio, out_sr, t_tts = tts.synthesize(translation, language=pack["target"])
    out_audio = os.path.join(REPO_ROOT, "processed",
                             f"language_eval_{pack['pair']}.wav")
    os.makedirs(os.path.dirname(out_audio), exist_ok=True)
    try:
        import soundfile as sf
        sf.write(out_audio, audio, out_sr)
    except Exception as exc:  # audio write is best-effort; the record still stands
        out_audio = f"(write failed: {exc})"

    record.update({
        "transcript": text,
        "translation": translation,
        "detected_lang": detected,
        "latency_s": {"stt": round(t_stt, 3), "tts": round(t_tts, 3),
                      "end_to_end": round(t_stt + t_tts, 3)},
        "audio_out": out_audio,
    })
    # Auto-fetch reference from speaker_voices.json if not provided in pack
    ref = args.ref or pack.get("reference")
    if not ref:
        try:
            meta = json.load(open(os.path.join(REPO_ROOT, "speaker_voices", "speaker_voices.json"), encoding="utf-8"))
            want = os.path.basename(clip)
            for e in meta:
                p = e.get("path", "") or e.get("filename", "") or ""
                if os.path.basename(p) == want:
                    ref = (e.get("transcribed_text") or "").strip()
                    break
        except Exception:
            pass
    # Compute WER if reference available
    wer_val = None
    if ref:
        try:
            from jiwer import wer
            wer_val = round(wer(ref, text), 4)
            record["wer"] = wer_val
            record["reference"] = ref
        except ImportError:
            record["wer"] = "jiwer not installed"
    # Determine status: verified only if WER < 0.2 (configurable threshold)
    # Lowered from 0.3 because jiwer normalization masks semantic errors
    # (Czech-like hallucinations can score deceptively low WER)
    WER_THRESHOLD = 0.2
    if wer_val is not None and wer_val < WER_THRESHOLD:
        record["status"] = "verified"
    else:
        record["status"] = "experimental"
        if wer_val is not None:
            record["reason"] = f"WER {wer_val:.3f} >= threshold {WER_THRESHOLD}"
        else:
            record["reason"] = "no reference available for quality verification"
    _write_eval(pack["pair"], record)
    print(f"{pack['pair']}: {record['status'].upper()}  stt {t_stt:.2f}s  tts {t_tts:.2f}s  clip {clip}")
    if wer_val is not None:
        print(f"  WER: {wer_val:.3f} (threshold {WER_THRESHOLD})")
    print(f"  transcript:  {text[:120]}")
    print(f"  translation: {translation[:120]}")
    print(f"  wrote {eval_path(pack['pair'])}")


def _write_eval(pair: str, record: dict) -> None:
    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(eval_path(pair), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------- entry

def main() -> None:
    parser = argparse.ArgumentParser(prog="language_packs", description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    p = sub.add_parser("show"); p.add_argument("pair")
    p = sub.add_parser("add"); p.add_argument("pair")
    p.add_argument("--license", default=None,
                   choices=["commercial-ok", "non-commercial", "unknown"],
                   help="override the licence inferred from the model card")
    p = sub.add_parser("remove"); p.add_argument("pair"); p.add_argument("--yes", action="store_true")
    p = sub.add_parser("test"); p.add_argument("pair")
    p.add_argument("--clip", default=None); p.add_argument("--ref", default=None)
    args = parser.parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    """Dispatch one action, turning library exceptions into a clean CLI exit."""
    try:
        {"list": cmd_list, "show": cmd_show, "add": cmd_add,
         "remove": cmd_remove, "test": cmd_test}[args.action](args)
    except (FileNotFoundError, FileExistsError, PermissionError, ValueError, KeyError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
