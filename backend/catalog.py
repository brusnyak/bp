"""Model catalog for the router feature.

Assembles a unified model list from:
  * curated entries (models/curated.json, tracked) — measured, recommended defaults
  * live cache (cache/catalog_live.json, git-ignored) — refreshed from HF/Piper

Computes hardware fit and per-stage/language recommendations. Never hides a model
for licence reasons; licence is an informational label only.

Usage:
    from backend.catalog import build_catalog, plan_route
    cat = build_catalog(hardware, stage="tts", lang="sk", q="")
    plan = plan_route("en", "sk", hardware)
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CURATED_PATH = os.path.join(REPO_ROOT, "models", "curated.json")
LIVE_CACHE_PATH = os.path.join(REPO_ROOT, "cache", "catalog_live.json")
USER_ENTRIES_PATH = os.path.join(REPO_ROOT, "config", "models.user.json")

# Default roots where installed model assets live. Discovered, not invented:
# these mirror the paths the existing code already uses (ct2_models, onnx_models,
# piper_models under backend/tts, plus a top-level models/ for router downloads).
MODEL_ROOTS = [
    os.path.join(REPO_ROOT, "ct2_models"),
    os.path.join(REPO_ROOT, "onnx_models"),
    os.path.join(REPO_ROOT, "backend", "tts", "piper_models"),
    os.path.join(REPO_ROOT, "models"),
]


def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def load_curated() -> list[dict]:
    data = _load_json(CURATED_PATH)
    if isinstance(data, dict) and isinstance(data.get("models"), list):
        return data["models"]
    if isinstance(data, list):
        return data
    return []


def load_live_cache() -> list[dict]:
    data = _load_json(LIVE_CACHE_PATH)
    if isinstance(data, dict) and isinstance(data.get("models"), list):
        return data["models"]
    if isinstance(data, list):
        return data
    return []


def load_user_entries() -> list[dict]:
    data = _load_json(USER_ENTRIES_PATH)
    if isinstance(data, dict) and isinstance(data.get("models"), list):
        return data["models"]
    if isinstance(data, list):
        return data
    return []


@dataclass
class Hardware:
    cpu_cores: int
    ram_gb: float
    gpu_backend: str        # 'cuda' | 'mps' | 'cpu' | 'none'
    gpu_name: Optional[str]
    vram_gb: Optional[float]
    disk_free_gb: float


def _fits(hw: Hardware, m: dict) -> str:
    """'good' | 'ok' | 'too-heavy' from hardware vs min_ram/min_vram/devices."""
    devices = m.get("devices") or ["cpu"]
    backend = hw.gpu_backend
    # Device compatibility: 'cpu' model runs on any backend; cuda/mps models need
    # that exact backend.
    if backend not in devices and "cpu" not in devices:
        return "too-heavy"
    if hw.ram_gb < (m.get("min_ram_gb") or 0):
        return "too-heavy"
    if backend in ("cuda", "mps") and hw.vram_gb is not None:
        if hw.vram_gb < (m.get("min_vram_gb") or 0):
            return "too-heavy"
    if m.get("min_vram_gb") and hw.vram_gb is None and backend in ("cuda", "mps"):
        return "ok"  # VRAM unknown: assume ok, flag as ok not good
    return "good"


def _recommended(models: list[dict]) -> list[dict]:
    """Best 'good' model per (stage, language|pair), measured numbers first."""
    best: dict[str, dict] = {}
    for m in models:
        if m.get("fit") != "good":
            continue
        key = (m.get("stage"), m.get("pair") or m.get("languages", [""])[0] if m.get("languages") else "")
        cur = best.get(key)
        if cur is None:
            best[key] = m
            continue
        # Prefer measured, then smaller, then recommended flag.
        def score(x: dict) -> tuple:
            return (
                0 if x.get("measured") else 1,
                x.get("size_mb") or 0,
                0 if x.get("recommended") else 1,
            )
        if score(m) < score(cur):
            best[key] = m
    return list(best.values())


def _installed(m: dict) -> bool:
    """Cheap file-based presence check under the known model roots."""
    files = m.get("source", {}).get("files") or {}
    # Piper-style: files.path maps voice key -> {size, md5}. Check any path exists.
    for _key, finfo in files.items():
        if isinstance(finfo, dict) and finfo.get("path"):
            path = finfo["path"]
            for root in MODEL_ROOTS:
                if os.path.exists(os.path.join(root, path)):
                    return True
    # Fallback: check id-derived dirs under roots.
    mid = m.get("id", "")
    for root in MODEL_ROOTS:
        for cand in (mid, mid.replace("/", "--"), mid.replace("/", "-")):
            if os.path.isdir(os.path.join(root, cand)) or os.path.isfile(os.path.join(root, cand)):
                return True
    return False


def build_catalog(hardware: dict, stage: Optional[str] = None,
                  lang: Optional[str] = None, q: Optional[str] = None) -> dict:
    """Assemble the full catalog: curated + live cache + user entries, merged by id."""
    curated = load_curated()
    live = load_live_cache()
    user = load_user_entries()

    by_id: dict[str, dict] = {}
    for m in curated + live + user:
        mid = m.get("id")
        if not mid:
            continue
        # Curated wins over live over user for conflicting ids.
        if mid not in by_id or m.get("source", {}).get("kind") == "curated":
            by_id[mid] = m

    models = list(by_id.values())
    hw = Hardware(
        cpu_cores=hardware.get("cpu_cores", 4),
        ram_gb=hardware.get("ram_gb", 8.0),
        gpu_backend=(hardware.get("gpu") or {}).get("backend", "cpu") if isinstance(hardware.get("gpu"), dict) else "cpu",
        gpu_name=(hardware.get("gpu") or {}).get("name") if isinstance(hardware.get("gpu"), dict) else None,
        vram_gb=(hardware.get("gpu") or {}).get("vram_gb") if isinstance(hardware.get("gpu"), dict) else None,
        disk_free_gb=hardware.get("disk_free_gb", 0.0),
    )

    out = []
    for m in models:
        mm = dict(m)
        mm["fit"] = _fits(hw, mm)
        mm["installed"] = _installed(mm)
        out.append(mm)

    # Filters
    if stage:
        out = [m for m in out if m.get("stage") == stage]
    if lang:
        out = [m for m in out if lang in (m.get("languages") or [])]
    if q:
        ql = q.lower()
        out = [m for m in out
               if ql in (m.get("name") or "").lower()
               or ql in (m.get("id") or "").lower()
               or ql in (m.get("engine") or "").lower()
               or ql in (m.get("pair") or "").lower()
               or any(ql in (c or "").lower() for c in (m.get("languages") or []))]

    rec_ids = {r["id"] for r in _recommended(out)}
    for m in out:
        m["recommended"] = m["id"] in rec_ids

    # Language index
    langs: dict[str, dict] = {}
    for m in out:
        for code in (m.get("languages") or []):
            entry = langs.setdefault(code, {"code": code, "name": code.upper(), "stt": 0, "mt_from": 0, "mt_to": 0, "tts": 0})
            st = m.get("stage")
            if st == "stt":
                entry["stt"] += 1
            elif st == "mt":
                pair = m.get("pair") or ""
                if pair.startswith(lang or code + "-"):
                    entry["mt_from"] += 1
                elif pair.endswith("-" + code):
                    entry["mt_to"] += 1
            elif st == "tts":
                entry["tts"] += 1

    return {
        "hardware": hardware,
        "models": out,
        "languages": sorted(langs.values(), key=lambda x: x["code"]),
        "source": {
            "curated": str(os.path.getmtime(CURATED_PATH)) if os.path.exists(CURATED_PATH) else None,
            "refreshed": str(os.path.getmtime(LIVE_CACHE_PATH)) if os.path.exists(LIVE_CACHE_PATH) else None,
        },
    }


def plan_route(src: str, tgt: str, hardware: dict) -> dict:
    """Best installed-or-installable chain for a language pair.

    Falls back to an English pivot (src->en->tgt) when no direct MT pair exists,
    flagged in the note. missing lists model ids not installed.
    """
    cat = build_catalog(hardware)
    models = cat["models"]

    def pick(stage: str, langs_filter: list[str]) -> Optional[dict]:
        cands = [m for m in models if m.get("stage") == stage
                 and m.get("fit") != "too-heavy"
                 and any(l in langs_filter for l in (m.get("languages") or []))]
        cands.sort(key=lambda m: (0 if m.get("recommended") else 1, m.get("size_mb") or 0))
        return cands[0] if cands else None

    stt = pick("stt", [src])
    tts = pick("tts", [tgt])
    direct = [m for m in models if m.get("stage") == "mt" and m.get("pair") == f"{src}-{tgt}"
              and m.get("fit") != "too-heavy"]
    direct.sort(key=lambda m: (0 if m.get("recommended") else 1, m.get("size_mb") or 0))
    mt = direct[0] if direct else None
    pivot = None
    note = None
    if mt is None:
        # English pivot: src->en then en->tgt. Recommend the en->tgt hop as the MT model.
        mt2 = pick("mt", [tgt]) if src != "en" else None
        mt = pick("mt", [tgt])
        pivot = True
        note = f"No direct {src}-{tgt} MT model; pivoting through English ({src}->en->{tgt})."
    else:
        pivot = False
        note = f"Direct {src}-{tgt} route."

    chain = {"stt": stt, "mt": mt, "tts": tts}
    missing = [m["id"] for m in chain.values() if m and not m.get("installed")]
    total_mb = sum((m.get("size_mb") or 0) for m in chain.values() if m)

    return {
        "src": src, "tgt": tgt,
        "stt": stt, "mt": mt, "tts": tts,
        "missing": missing,
        "total_mb": total_mb,
        "pivot": pivot,
        "note": note or f"Route {src}->{tgt} planned.",
    }
