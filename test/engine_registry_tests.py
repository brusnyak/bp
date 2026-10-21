"""Pins for the engine registry + licence gate (spec 003, T007/T008).

The load-bearing invariant: every engine the live pipeline can actually build MUST be
registered, so no engine runs un-gated. The rest pins the fail-closed gate behaviour.
"""
import pytest

import backend.registry as reg
from backend.tts.base import TTS_ENGINES


def test_every_live_tts_engine_is_registered():
    """Drift pin: TTS_ENGINES is the live path (main.py). Anything buildable there must
    have a manifest here, or the gate silently describes a different set of engines."""
    registered = {c.name for c in reg.engines("tts")}
    missing = set(TTS_ENGINES) - registered
    assert not missing, f"live TTS engines with no registry manifest: {sorted(missing)}"


def test_gate_blocks_noncommercial_by_default():
    for stage, name in (("tts", "xtts"), ("mt", "nllb")):
        cap = reg.get(stage, name)
        if cap.license != "non-commercial":       # nllb factory may be absent, label still stands
            continue
        with pytest.raises(PermissionError):
            reg.select(stage, name)
        assert reg.select(stage, name, allow_noncommercial=True) is cap


def test_gate_fails_closed_on_unknown_licence():
    cap = reg.get("tts", "omnivoice")
    assert cap.license == "unknown"
    with pytest.raises(PermissionError):
        reg.select("tts", "omnivoice")
    assert reg.select("tts", "omnivoice", allow_unknown=True) is cap


def test_gate_allows_commercial_engines():
    cap = reg.select("tts", "piper")
    assert cap.license == "commercial-ok"


def test_declared_only_engine_is_not_available():
    cap = reg.get("unified", "qwen3-omni")
    available, reason = cap.check()
    assert available is False and reason  # a reason, not a silent False


def test_instantiate_refuses_declared_only():
    with pytest.raises(RuntimeError):
        reg.instantiate("unified", "qwen3-omni")


def test_duplicate_registration_is_rejected():
    cap = reg.get("tts", "piper")
    with pytest.raises(ValueError):
        reg.register(cap)


def test_probe_error_is_surfaced_not_hidden():
    cap = reg.Capability("boom", "tts", factory=lambda: object(),
                         probe=lambda: (_ for _ in ()).throw(ValueError("kaboom")))
    available, reason = cap.check()
    assert available is False
    assert reason.startswith("probe error") and "kaboom" in reason


def test_catalog_covers_all_stages():
    cat = reg.catalog()
    assert set(cat) == set(reg.STAGES)
    assert cat["tts"] and cat["mt"]
