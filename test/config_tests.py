import backend.main as bm


def test_slovak_prefers_local_tuned_model_then_turbo(monkeypatch, tmp_path):
    monkeypatch.delenv("BP_SK_STT_MODEL", raising=False)
    monkeypatch.setattr(bm, "SK_STT_LOCAL_MODEL", str(tmp_path / "missing"))
    for requested in ("base", "small"):
        assert bm._pick_stt_model("sk", requested) == "large-v3-turbo"  # fallback when the tuned model is not converted
    local = tmp_path / "whisper-small-sk"; local.mkdir(); (local / "model.bin").write_bytes(b"x")
    monkeypatch.setattr(bm, "SK_STT_LOCAL_MODEL", str(local))
    assert bm._pick_stt_model("sk", "base") == str(local)


def test_sk_stt_model_env_wins(monkeypatch, tmp_path):
    local = tmp_path / "m"; local.mkdir(); (local / "model.bin").write_bytes(b"x")
    monkeypatch.setattr(bm, "SK_STT_LOCAL_MODEL", str(local))
    monkeypatch.setenv("BP_SK_STT_MODEL", "small")
    assert bm._pick_stt_model("sk", "base") == "small"


def test_other_languages_and_explicit_models_are_untouched(monkeypatch):
    monkeypatch.delenv("BP_SK_STT_MODEL", raising=False)
    assert bm._pick_stt_model("en", "base") == "base"
    assert bm._pick_stt_model("sk", "medium") == "medium"


def test_whisper_prefers_local_model_directory(monkeypatch, tmp_path):
    import backend.stt.faster_whisper_stt as fw
    seen = []
    monkeypatch.setattr(fw, "WhisperModel", lambda source, **kw: seen.append(source))
    monkeypatch.setattr(fw, "LOCAL_MODELS_DIR", str(tmp_path))
    fw.FasterWhisperSTT(model_size="base")
    (tmp_path / "whisper-base").mkdir(); (tmp_path / "whisper-base" / "model.bin").write_bytes(b"x")
    fw.FasterWhisperSTT(model_size="base")
    assert seen == ["base", str(tmp_path / "whisper-base")]  # cache name when absent, plain directory when fetched
