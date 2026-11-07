"""Model-free test for scripts/setup.py language/profile planning."""
import importlib.util, pathlib

spec = importlib.util.spec_from_file_location("bp_setup", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "setup.py")
setup = importlib.util.module_from_spec(spec); spec.loader.exec_module(setup)


def test_german_plan_has_voice_and_both_directions():
    voices, pairs, pip = setup.plan("de", None)
    assert voices == ["en_US-john-medium", "de_DE-thorsten-medium"] and pairs == ["en-de", "de-en"] and pip == []


def test_polish_is_one_directional_and_profile_adds_engine():
    _, pairs, pip = setup.plan("pl", "mac")
    assert pairs == ["pl-en"] and "mlx-whisper" in pip


def test_unknown_language_is_rejected():
    try:
        setup.plan("xx", None)
    except SystemExit as e:
        assert "unknown language" in str(e)
    else:
        raise AssertionError("expected SystemExit")


def test_every_preset_file_exists_for_every_profile():
    root = pathlib.Path(__file__).resolve().parents[1]
    for name in setup.PROFILES:
        assert (root / "presets" / f"{name}.env").is_file(), name
