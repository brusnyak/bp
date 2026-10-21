"""Engine registry — one place that answers: which engines exist per stage, what
each one can do, and which ones may ship.

This does NOT rewrite the pipeline. Existing classes and factories stay exactly as
they are; they are described here with a capability manifest and passed through a
licence gate. Selection by config (T010) and gate tests (T009) build on this.

    from backend.registry import engines, select, instantiate, catalog
    cap = instantiate("tts", "piper")            # gate + availability + build
    cap = select("tts", "xtts", allow_noncommercial=True)

Licence policy (documentation/s2s_landscape_2026-10-18.md §8) — FAIL CLOSED:
  commercial-ok    ships in the default profile
  non-commercial   blocked unless allow_noncommercial=True  (XTTS v2, NLLB)
  unknown          blocked unless allow_unknown=True         (confirm the model card first)

`select` answers "may this ship"; `instantiate` additionally requires the engine to be
installed here. Both raise instead of silently passing an unlabelled engine.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)

STAGES = ("vad", "stt", "mt", "tts", "unified")
LICENSES = ("commercial-ok", "non-commercial", "unknown")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_repo_on_path() -> None:
    """Make `import backend...` work regardless of the process CWD."""
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)


@dataclass(frozen=True)
class Capability:
    name: str
    stage: str
    factory: Optional[Callable[[], object]] = None
    probe: Optional[Callable[[], bool]] = None   # cheap availability check, no model load
    languages: Optional[tuple] = None            # None = language comes from the pack, not the engine
    devices: tuple = ("cpu",)
    streaming: bool = False
    license: str = "unknown"
    license_note: str = ""
    resource: str = ""
    notes: str = ""

    def check(self) -> "tuple[bool, Optional[str]]":
        """(installed-here?, reason-if-not). A probe that *errors* is reported as an
        error, never silently folded into 'not installed'."""
        if self.factory is None:
            return False, self.notes or "declared only — no factory yet"
        if self.probe is None:
            return True, None
        try:
            return (True, None) if self.probe() else (False, "not installed here")
        except Exception as exc:  # surface the bug instead of hiding it
            return False, f"probe error: {type(exc).__name__}: {exc}"

    def available(self) -> bool:
        """Installed *and* constructible here. A declared-but-unbuilt engine (factory
        is None) is never 'available'."""
        return self.check()[0]

    def describe(self) -> dict:
        ok, reason = self.check()
        return {
            "name": self.name, "stage": self.stage, "available": ok,
            "unavailable_reason": reason,
            "languages": list(self.languages) if self.languages else None,
            "devices": list(self.devices), "streaming": self.streaming,
            "license": self.license, "license_note": self.license_note,
            "resource": self.resource, "notes": self.notes,
        }


_REGISTRY: dict[tuple[str, str], Capability] = {}


def register(cap: Capability) -> Capability:
    if cap.stage not in STAGES:
        raise ValueError(f"unknown stage '{cap.stage}' (want one of {STAGES})")
    if cap.license not in LICENSES:
        raise ValueError(f"unknown license '{cap.license}' (want one of {LICENSES})")
    key = (cap.stage, cap.name)
    if key in _REGISTRY:
        raise ValueError(f"engine '{cap.name}' already registered for stage '{cap.stage}'")
    _REGISTRY[key] = cap
    return cap


def engines(stage: Optional[str] = None) -> list[Capability]:
    return [c for (s, _), c in sorted(_REGISTRY.items()) if stage is None or s == stage]


def get(stage: str, name: str) -> Capability:
    try:
        return _REGISTRY[(stage, name)]
    except KeyError:
        known = ", ".join(c.name for c in engines(stage)) or "none"
        raise KeyError(f"no engine '{name}' in stage '{stage}' (known: {known})") from None


def license_ok(cap: Capability, allow_noncommercial: bool = False,
               allow_unknown: bool = False) -> bool:
    if cap.license == "commercial-ok":
        return True
    if cap.license == "non-commercial":
        return allow_noncommercial
    return allow_unknown  # unknown: fail closed


def select(stage: str, name: str, allow_noncommercial: bool = False,
           allow_unknown: bool = False) -> Capability:
    """Resolve an engine for use, enforcing the licence gate (fail closed)."""
    cap = get(stage, name)
    if not license_ok(cap, allow_noncommercial, allow_unknown):
        hint = ("pass allow_noncommercial=True only for internal/non-shipping runs"
                if cap.license == "non-commercial"
                else "confirm the model-card licence, then pass allow_unknown=True")
        raise PermissionError(
            f"engine '{name}' ({stage}) is {cap.license}: "
            f"{cap.license_note or 'no licence note'} — {hint}")
    if cap.license != "commercial-ok":
        logger.warning("registry: '%s' (%s) passed the gate as %s by explicit override — "
                       "not shippable", name, stage, cap.license)
    return cap


def instantiate(stage: str, name: str, allow_noncommercial: bool = False,
                allow_unknown: bool = False, **factory_kwargs) -> object:
    """Gate, then check the engine is installed here, then build it.
    Any extra kwargs are passed to the factory callable."""
    cap = select(stage, name, allow_noncommercial, allow_unknown)
    if cap.factory is None:
        raise RuntimeError(f"engine '{name}' ({stage}) is declared but not implemented")
    if not cap.available():
        raise RuntimeError(f"engine '{name}' ({stage}) is registered but not installed here")
    return cap.factory(**factory_kwargs)


def catalog() -> dict:
    return {s: [c.describe() for c in engines(s)] for s in STAGES}


# --------------------------------------------------------------------------
# The real engines, described. Lazy probes so importing this module is cheap
# and torch-free installs still work.

def _has_module(mod: str) -> Callable[[], bool]:
    def probe() -> bool:
        import importlib.util
        _ensure_repo_on_path()
        return importlib.util.find_spec(mod) is not None
    return probe


def _has_file(relpath: str) -> Callable[[], bool]:
    def probe() -> bool:
        return os.path.exists(os.path.join(_REPO_ROOT, relpath))
    return probe


def _has_dir(relpath: str) -> Callable[[], bool]:
    def probe() -> bool:
        return os.path.isdir(os.path.join(_REPO_ROOT, relpath))
    return probe


def _in_tts(key: str) -> Callable[[], bool]:
    """Availability probe for a TTS engine: present in the (guarded) TTS_ENGINES map."""
    def probe() -> bool:
        _ensure_repo_on_path()
        from backend.tts.base import TTS_ENGINES
        return key in TTS_ENGINES
    return probe


def _tts_factory(key: str) -> Callable[[], object]:
    def make():
        _ensure_repo_on_path()
        from backend.tts.base import TTS_ENGINES
        return TTS_ENGINES[key]()
    return make


def _stt_factory(model_size: str = "base") -> Callable[..., object]:
    def make(**kwargs):
        _ensure_repo_on_path()
        from backend.stt.faster_whisper_stt import FasterWhisperSTT
        size = kwargs.get("model_size", model_size)
        return FasterWhisperSTT(model_size=size)
    return make


def _mt_factory(model_path: str = "Helsinki-NLP--opus-mt-en-sk") -> Callable[..., object]:
    def make(**kwargs):
        _ensure_repo_on_path()
        from backend.mt.ctranslate2_mt import CTranslate2MT
        path = kwargs.get("model_path", model_path)
        return CTranslate2MT(model_path=path, device=kwargs.get("device", "cpu"))
    return make


def _nllb_factory() -> Callable[..., object]:
    def make(**kwargs):
        _ensure_repo_on_path()
        from backend.mt.nllb_mt import NLLBMT
        return NLLBMT()
    return make


def _hybrid_mt_factory() -> Callable[..., object]:
    def make(**kwargs):
        _ensure_repo_on_path()
        from backend.mt.hybrid_mt import HybridMT
        return HybridMT()
    return make


def rvc_mlx_factory(**kwargs) -> object:
    _ensure_repo_on_path()
    from backend.tts.rvc_mlx import RVCMLX, RVCConfig
    config = RVCConfig(**kwargs)
    return RVCMLX(config)


def rvc_onnx_factory(**kwargs) -> object:
    _ensure_repo_on_path()
    from backend.tts.rvc_mlx import RVCOnnx, RVCConfig
    config = RVCConfig(**kwargs)
    return RVCOnnx(config)


def _combine(*probes: Callable[[], bool]) -> Callable[[], bool]:
    def probe() -> bool:
        return all(p() for p in probes)
    return probe


def _register_defaults() -> None:
    # VAD ------------------------------------------------------------------
    register(Capability(
        "webrtcvad", "vad", factory=lambda: __import__("webrtcvad").Vad(2),
        probe=_has_module("webrtcvad"), devices=("cpu",), streaming=True,
        license="commercial-ok", license_note="webrtcvad is BSD-3-Clause",
        resource="~1 MB",
        notes="default live gate; the live path sets its own aggressiveness "
              "(VAD_AGGRESSIVENESS in backend/main.py); Vad(2) here is the standalone default"))
    register(Capability(
        "silero", "vad", factory=None,
        probe=_combine(_has_module("torch"),
                       _has_file("backend/stt/whisper_streaming_vendor/silero_vad_iterator.py")),
        devices=("cpu",), streaming=True, license="commercial-ok",
        license_note="Silero VAD is MIT (snakers4/silero-vad)",
        resource="~2 MB + torch",
        notes="used through the streaming STT path, not a standalone class "
              "(whisper_streaming_vendor/whisper_online.py loads it via torch.hub)"))

    # STT ------------------------------------------------------------------
    register(Capability(
        "faster_whisper", "stt", factory=_stt_factory("base"),
        probe=_has_module("faster_whisper"), devices=("cpu", "cuda", "directml", "coreml"),
        streaming=False, license="commercial-ok",
        license_note="faster-whisper + CTranslate2 MIT; whisper weights MIT",
        resource="~1 GB per loaded model (int8 smaller)",
        notes="chunked, not streaming; model_size is per-language-pack"))

    # MT -------------------------------------------------------------------
    register(Capability(
        "opus-mt", "mt", factory=_mt_factory(),
        probe=_combine(_has_module("ctranslate2"), _has_dir("ct2_models")),
        devices=("cpu", "cuda"), streaming=False, license="commercial-ok",
        license_note="Helsinki-NLP Opus-MT is per-model (mostly CC-BY-4.0) — verify per pair",
        resource="~300 MB per direction (int8)",
        notes="one model per direction; the default MT; translates a segment at a time. "
              "availability needs BOTH ctranslate2 and converted weights in ct2_models/; "
              "instantiate still fails for a pair that was never converted"))
    register(Capability(
        "nllb", "mt", factory=_nllb_factory(), probe=_has_module("transformers"),
        devices=("cpu", "cuda"), streaming=False, license="non-commercial",
        license_note="NLLB-200 is CC-BY-NC-4.0 (non-commercial)",
        resource="~2.5 GB", notes="200 languages, but not shippable"))
    register(Capability(
        "hybrid-mt", "mt", factory=_hybrid_mt_factory(), probe=_has_module("transformers"),
        devices=("cpu", "cuda"), streaming=False, license="unknown",
        license_note="inherits component licences (verify the components before shipping)",
        resource="varies", notes="backend/mt/hybrid_mt.py"))

    # TTS ------------------------------------------------------------------
    register(Capability(
        "piper", "tts", factory=_tts_factory("piper"), probe=_in_tts("piper"),
        devices=("cpu",), streaming=True, license="commercial-ok",
        license_note="Piper is MIT", resource="~60 MB per voice",
        notes="CPU baseline; the voice file under backend/tts/piper_models/ is per-voice "
              "and local-only, so the probe says 'installed' before the voice is guaranteed"))


def _register_optional() -> list:
    """Engines whose module may be absent in a torch-free install. Each entry is
    guarded on its own, so one bad entry cannot silently drop the rest."""
    entries = []
    for name, lang, notes in (("piper_personal", ("en",), "EN fine-tune (local-only voice)"),
                              ("piper_personal_v2", ("en",), "EN fine-tune v2 (local-only)"),
                              ("piper_sk_personal", ("sk",), "SK fine-tune (local-only)"),
                              ("piper_sk_omni", ("sk",), "SK OmniVoice-Piper hybrid voice"),
                              ("piper_generic_sk", ("sk",), "generic SK Piper voice"),
                              ("piper_generic_en", ("en",), "generic EN Piper voice")):
        entries.append(Capability(
            name, "tts", factory=_tts_factory(name), probe=_in_tts(name),
            languages=lang, devices=("cpu",), streaming=True, license="commercial-ok",
            license_note="Piper is MIT", resource="~60 MB per voice",
            notes=notes + " (voice file is local-only; falls back to the generic voice if absent)"))
    entries.append(Capability(
        "hybrid", "tts", factory=_tts_factory("hybrid"), probe=_in_tts("hybrid"),
        devices=("cpu", "cuda", "mps"), streaming=True, license="unknown",
        license_note="inherits component licences (if XTTS is used => non-commercial)",
        resource="varies", notes="backend/tts/hybrid_tts.py (SUPPORTS_STREAMING=True)"))
    entries.append(Capability(
        "omnivoice", "tts", factory=_tts_factory("omnivoice"), probe=_in_tts("omnivoice"),
        devices=("cuda", "mps", "cpu"), streaming=True, license="unknown",
        license_note="confirm the model card licence before shipping",
        resource="~5.8 GB VRAM at 12 steps (T4)",
        notes="cloning; SUPPORTS_STREAMING=True; needs a GPU for real use"))
    entries.append(Capability(
        "xtts", "tts", factory=_tts_factory("xtts"), probe=_in_tts("xtts"),
        devices=("cuda", "mps"), streaming=True, license="non-commercial",
        license_note="Coqui XTTS v2 is under the Coqui Public Model License (non-commercial)",
        resource="~2 GB", notes="cloning; not shippable"))
    # unified (declared, not implemented — FR-014) --------------------------
    entries.append(Capability(
        "qwen3-omni", "unified", factory=None, probe=None, devices=("cuda",),
        streaming=True, license="commercial-ok", license_note="Qwen3-Omni is Apache-2.0",
        resource="~30B params", notes="planned adapter (spec 003 T025); not built yet"))
    # Voice conversion layer (engine-agnostic, MIT) ----------------------------
    entries.append(Capability(
        "rvc-mlx", "tts", factory=rvc_mlx_factory, probe=_has_module("rvc_mlx"),
        devices=("mps", "cpu"), streaming=False, license="commercial-ok",
        license_note="RVC-MLX is MIT (Acelogic/Retrieval-based-Voice-Conversion-MLX)",
        resource="~500 MB model", notes="Engine-agnostic voice layer: train once, any TTS → your voice. Mac only (MLX)."))
    entries.append(Capability(
        "rvc-onnx", "tts", factory=rvc_onnx_factory, probe=_has_module("onnxruntime"),
        devices=("cpu", "cuda"), streaming=False, license="commercial-ok",
        license_note="onyx is MIT (hixady/onyx)", resource="~500 MB model",
        notes="Cross-platform RVC via ONNX Runtime. Linux/Windows/CPU/CUDA. Pack .pth + .index → .onyx first."))

    skipped = []
    for cap in entries:
        try:
            register(cap)
        except Exception as exc:  # one bad entry must not drop the others
            logger.warning("registry: skipped %s/%s: %s", cap.stage, cap.name, exc)
            skipped.append((cap.name, str(exc)))
    return skipped


def skipped_optional() -> list:
    """Optional engines that failed to register at import (name, reason). Empty is normal."""
    return list(_OPTIONAL_SKIPPED)


_register_defaults()
_OPTIONAL_SKIPPED = _register_optional()
