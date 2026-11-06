"""Multi-language evidence tests for scripts/language_packs.

Stdlib only (unittest, tempfile, importlib, unittest.mock, json, shutil).
No fastapi, no torch, no faster-whisper, no model loading, no network.

These tests exercise the language-pack layer across every declared pair
(en-sk, sk-en, en-cs, cs-en, en-de, en-pl, en-zh, fr-en, hi-en, ja-en,
pl-en, en-hi, zh-en, de-en, en-fr) — not just Slovak/English — so adding a
language is a matter of dropping a JSON pack into languages/packs/, not editing code.

A pack is "verified" only when its components resolve AND a recorded eval
record exists; otherwise it is "experimental". These tests assert the
experimental default (clean clone has no ct2_models/ or eval records) and
that resolve/status behave per-pack.
"""
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE_PATH = os.path.join(REPO_ROOT, "scripts", "language_packs.py")

REAL_PACKS_DIR = os.path.join(REPO_ROOT, "languages", "packs")

EXPECTED_PAIRS = {
    "en-sk", "sk-en", "en-cs", "cs-en", "en-de", "en-pl", "en-zh",
    "fr-en", "hi-en", "ja-en", "pl-en", "en-hi", "zh-en", "de-en", "en-fr",
}


def load_module():
    spec = importlib.util.spec_from_file_location("language_packs_under_test", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class HermeticEnv:
    """Temp copy of the repo root where language_packs reads/writes nothing real."""

    def __init__(self):
        self.tmp = tempfile.mkdtemp(prefix="bp_langtest_")
        self.packs = os.path.join(self.tmp, "languages", "packs")
        self.ct2 = os.path.join(self.tmp, "ct2_models")
        self.piper = os.path.join(self.tmp, "piper_models")
        self.eval = os.path.join(self.tmp, "processed", "language_eval")
        for d in (self.packs, self.ct2, self.piper, self.eval):
            os.makedirs(d, exist_ok=True)
        # Copy the real declared packs so tests run over all 13 pairs.
        for name in os.listdir(REAL_PACKS_DIR):
            if name.endswith(".json"):
                shutil.copy(os.path.join(REAL_PACKS_DIR, name), os.path.join(self.packs, name))

    def apply(self, mod):
        mod.REPO_ROOT = self.tmp
        mod.PACKS_DIR = self.packs
        mod.CT2_DIR = self.ct2
        mod.PIPER_DIR = self.piper
        mod.EVAL_DIR = self.eval

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class LanguagePacksMultiLanguageTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_module()
        self.env = HermeticEnv()
        self.env.apply(self.mod)

    def tearDown(self):
        self.env.cleanup()

    def test_module_loads_without_ml_or_web_deps(self):
        mod = self.mod
        for name in ("load", "all_packs", "resolve", "status", "pack_path",
                     "resolve_stt", "resolve_mt", "resolve_tts", "cmd_add",
                     "cmd_remove", "hf_model_info", "eval_path"):
            self.assertTrue(hasattr(mod, name), name)

    def test_all_declared_packs_load(self):
        mod = self.mod
        packs = mod.all_packs()
        self.assertEqual(len(packs), len(EXPECTED_PAIRS))
        got = {p["pair"] for p in packs}
        self.assertEqual(got, EXPECTED_PAIRS)
        for pack in packs:
            for key in ("pair", "source", "target", "stt", "mt", "tts"):
                self.assertIn(key, pack, f"{pack.get('pair')} missing {key}")

    def test_pack_path_resolves(self):
        mod = self.mod
        self.assertEqual(mod.pack_path("en-sk"),
                         os.path.join(self.env.packs, "en-sk.json"))

    def test_resolve_returns_full_structure_for_every_pack(self):
        mod = self.mod
        for pack in mod.all_packs():
            r = mod.resolve(pack)
            self.assertIn("stt", r)
            self.assertIn("mt", r)
            self.assertIn("tts", r)
            self.assertIn("errors", r)
            self.assertIn("resolved", r)
            self.assertIsInstance(r["errors"], list)

    def test_status_is_experimental_without_eval_records(self):
        mod = self.mod
        # Clean clone: no ct2_models/, no piper_models/, no eval records ->
        # every pack is experimental, none verified.
        for pack in mod.all_packs():
            self.assertEqual(mod.status(pack), "experimental",
                             f"{pack['pair']} should be experimental on a clean clone")

    def test_stt_resolve_known_sizes_available(self):
        mod = self.mod
        # "base" and "small" are in the HF fallback whisper set, so they resolve
        # even without a local ct2_models/ directory.
        for model in ("base", "small"):
            stt, err = mod.resolve_stt({"model": model, "language": "en"})
            self.assertEqual(stt, model, f"STT {model} should resolve: {err}")
            self.assertIsNone(err)

    def test_stt_resolve_unknown_size_is_error(self):
        mod = self.mod
        stt, err = mod.resolve_stt({"model": "no-such-size", "language": "en"})
        self.assertIsNone(stt)
        self.assertIsNotNone(err)

    def test_mt_requires_conversion_dir(self):
        mod = self.mod
        mt, err = mod.resolve_mt({"model": "Helsinki-NLP--opus-mt-en-sk"})
        self.assertIsNone(mt)
        self.assertIsNotNone(err)
        self.assertIn("converted", err)

    def test_tts_requires_local_voice(self):
        mod = self.mod
        tts, err = mod.resolve_tts({"voice": "sk_SK-lili-medium", "fallback": None})
        self.assertIsNone(tts)
        self.assertIsNotNone(err)

    def test_unresolved_pack_never_verified(self):
        mod = self.mod
        pack = mod.load("en-sk")
        r = mod.resolve(pack)
        self.assertFalse(r["resolved"])
        self.assertEqual(mod.status(pack), "experimental")

    def test_cmd_add_writes_pack_and_reports_experimental(self):
        mod = self.mod
        args = type("A", (), {"pair": "pt-en", "license": None})()
        with patch.object(mod, "hf_model_info", return_value=(True, "commercial-ok")):
            mod.cmd_add(args)
        written = os.path.join(self.env.packs, "pt-en.json")
        self.assertTrue(os.path.isfile(written), "cmd_add should write the pack file")
        with open(written, encoding="utf-8") as f:
            written_pack = json.load(f)
        self.assertEqual(written_pack["pair"], "pt-en")
        # New pack with no components resolved and no eval record is experimental.
        self.assertEqual(mod.status(written_pack), "experimental")

    def test_cmd_remove_deletes_pack(self):
        mod = self.mod
        args = type("A", (), {"pair": "en-sk", "yes": True})()
        mod.cmd_remove(args)
        self.assertFalse(os.path.isfile(mod.pack_path("en-sk")))
        self.assertNotIn("en-sk", {p["pair"] for p in mod.all_packs()})

    def test_hf_model_info_returns_tuple(self):
        mod = self.mod
        # Offline / unauthenticated HF lookups degrade gracefully, never raise.
        with patch.object(mod.urllib.request, "urlopen", side_effect=OSError("offline")):
            ok, lic = mod.hf_model_info("Helsinki-NLP/opus-mt-en-sk", timeout=2.0)
        self.assertEqual((ok, lic), (True, "unknown"))


if __name__ == "__main__":
    unittest.main()
