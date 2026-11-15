"""Catalog + route-plan tests for the router feature (no server, no downloads)."""
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from backend.catalog import build_catalog, load_curated, plan_route, CURATED_PATH  # noqa: E402


FAKE_HW = {
    "cpu_cores": 4,
    "ram_gb": 8.0,
    "gpu": {"name": None, "vram_gb": None, "backend": "cpu"},
    "disk_free_gb": 100.0,
}
FAKE_HW_GPU = {
    "cpu_cores": 8,
    "ram_gb": 16.0,
    "gpu": {"name": "T4", "vram_gb": 16.0, "backend": "cuda"},
    "disk_free_gb": 100.0,
}


def test_curated_has_min_models():
    models = load_curated()
    assert len(models) >= 40, f"expected >= 40 curated models, got {len(models)}"


def test_catalog_shape_and_filters():
    cat = build_catalog(FAKE_HW)
    assert "hardware" in cat and "models" in cat and "languages" in cat
    assert isinstance(cat["models"], list) and len(cat["models"]) > 0
    # every model has required keys
    for m in cat["models"]:
        for k in ("id", "stage", "name", "engine", "fit", "installed", "recommended"):
            assert k in m, f"missing key {k} in {m.get('id')}"
        assert m["fit"] in ("good", "ok", "too-heavy")
    # stage filter
    stt = build_catalog(FAKE_HW, stage="stt")["models"]
    assert all(m["stage"] == "stt" for m in stt)
    # lang filter
    sk = build_catalog(FAKE_HW, lang="sk")["models"]
    assert all("sk" in (m.get("languages") or []) for m in sk)
    # q filter
    q = build_catalog(FAKE_HW, q="whisper")["models"]
    assert all("whisper" in (m["name"] + m["id"] + m["engine"]).lower() for m in q)


def test_fit_too_heavy_on_cpu_for_gpu_only():
    cat = build_catalog(FAKE_HW)
    higgs = next(m for m in cat["models"] if m["id"] == "higgs-audio-v3-4b")
    # cpu-only, gpu-only model => too-heavy
    assert higgs["fit"] == "too-heavy"


def test_recommended_best_good_per_stage_lang():
    cat = build_catalog(FAKE_HW)
    tts_good = [m for m in cat["models"] if m["stage"] == "tts" and m["fit"] == "good"]
    recs = [m for m in tts_good if m["recommended"]]
    # at least one recommended good tts per language family
    langs = set()
    for m in recs:
        langs.update(m.get("languages") or [])
    assert "en" in langs


def test_plan_route_en_sk():
    plan = plan_route("en", "sk", FAKE_HW)
    assert plan["src"] == "en" and plan["tgt"] == "sk"
    assert plan["stt"] is not None and plan["mt"] is not None and plan["tts"] is not None
    assert plan["mt"]["pair"] == "en-sk"
    assert "missing" in plan and isinstance(plan["missing"], list)
    assert isinstance(plan["total_mb"], (int, float))


def test_plan_route_pivot_when_no_direct():
    # pair with no direct MT -> pivot flagged
    plan = plan_route("fr", "sk", FAKE_HW)
    assert plan["pivot"] is True
    assert "pivot" in plan["note"].lower() or "english" in plan["note"].lower()


def test_endpoints_via_testclient():
    from fastapi.testclient import TestClient
    from app import create_app
    from backend.utils.auth import GuestSession
    from backend.main import get_current_session
    app = create_app()
    app.dependency_overrides[get_current_session] = lambda: GuestSession("0" * 32)
    client = TestClient(app)

    r = client.get("/api/catalog")
    assert r.status_code == 200
    body = r.json()
    assert "models" in body and "languages" in body

    r = client.get("/api/catalog?stage=tts&lang=en")
    assert r.status_code == 200
    assert all(m["stage"] == "tts" for m in r.json()["models"])

    r = client.get("/api/hardware")
    assert r.status_code == 200
    assert "cpu_cores" in r.json()

    r = client.get("/api/route/plan?src=en&tgt=sk")
    assert r.status_code == 200
    assert r.json()["mt"]["pair"] == "en-sk"

    r = client.get("/api/route")
    assert r.status_code == 200
    assert "src" in r.json() and "tgt" in r.json()

    r = client.post("/api/route", json={"src": "en", "tgt": "sk", "stt": "x"})
    assert r.status_code == 200
    assert r.json()["src"] == "en"

    r = client.post("/api/models/install", json={"ids": ["piper-en_US-lessac-medium"]})
    assert r.status_code == 200
    assert "job_id" in r.json()
    job = r.json()["job_id"]
    r2 = client.get(f"/api/jobs/{job}")
    assert r2.status_code == 200
    assert r2.json()["state"] in ("queued", "running", "done", "error")

    r = client.delete("/api/models/nope-not-installed")
    assert r.status_code == 404
    app.dependency_overrides.clear()
