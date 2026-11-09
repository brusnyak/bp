from backend.main import STRATEGIES, _apply_strategy


def test_each_strategy_sets_its_cutting_rules():
    for name, preset in STRATEGIES.items():
        cfg = {"strategy": name, "vad_silence_timeout_s": 9, "vad_max_segment_s": 9}
        _apply_strategy(cfg)
        assert all(cfg[k] == v for k, v in preset.items()), name
    assert STRATEGIES["chunks"]["stream_chunks"] and not STRATEGIES["phrase"]["stream_chunks"]
    assert STRATEGIES["agree"]["incremental"] and not STRATEGIES["clause"]["incremental"]


def test_unknown_or_missing_strategy_keeps_env_defaults():
    cfg = {"strategy": "nope", "vad_max_segment_s": 3.0}
    _apply_strategy(cfg); assert cfg == {"strategy": "nope", "vad_max_segment_s": 3.0}
    cfg = {"vad_max_segment_s": 3.0}
    _apply_strategy(cfg); assert cfg == {"vad_max_segment_s": 3.0}


def test_agree_short_sets_fragment_sizes_and_other_strategies_clear_them():
    cfg = {"strategy": "agree_short"}; _apply_strategy(cfg)
    assert cfg["incr_min_words"] == 2 and cfg["incr_max_wait_s"] == 0.8 and cfg["incremental"]
    cfg["strategy"] = "phrase"; _apply_strategy(cfg)
    assert "incr_min_words" not in cfg and not cfg["incremental"]


def test_default_strategy_follows_env_and_rejects_unknown(monkeypatch):
    from backend.main import default_strategy
    monkeypatch.delenv("BP_DEFAULT_STRATEGY", raising=False)
    assert default_strategy() == "phrase"
    monkeypatch.setenv("BP_DEFAULT_STRATEGY", "agree_short")
    assert default_strategy() == "agree_short"
    monkeypatch.setenv("BP_DEFAULT_STRATEGY", "nonsense")
    assert default_strategy() == "phrase"
