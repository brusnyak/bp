"""Every language pack's target must resolve to a registered TTS engine (2026-11-07: German crashed live with 'no engine piper_generic_de')."""
import glob, json, os
from backend.registry import get, target_tts_engine


def test_every_pack_target_has_a_registered_voice_engine():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    missing = []
    for path in sorted(glob.glob(os.path.join(root, "languages", "packs", "*.json"))):
        tgt = os.path.basename(path)[:-5].split("-")[1]
        engine = target_tts_engine("piper", tgt)
        try:
            get("tts", engine)
        except KeyError:
            missing.append((os.path.basename(path), engine))
    assert not missing, missing
