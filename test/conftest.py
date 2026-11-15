"""Shared API fixtures: anonymous sessions and isolated local recordings."""
import os
os.environ.setdefault("JWT_SECRET", "test-only-secret-not-used-outside-pytest")
os.environ.setdefault("BP_DISABLE_RATE_LIMIT", "1")
import pytest
from fastapi.testclient import TestClient
from app import create_app

@pytest.fixture(autouse=True)
def isolated_references(tmp_path, monkeypatch):
    import backend.main as bm
    directory = tmp_path / "references"
    directory.mkdir()
    monkeypatch.setattr(bm, "VOICES_DIR", str(directory))

@pytest.fixture
def test_client():
    with TestClient(create_app()) as client:
        yield client
