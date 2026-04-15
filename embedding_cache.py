"""Embedding cache – load, persist, and manage sorted video embedding caches."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TypedDict

from shared.constants import (
    LEARNING_DIRNAME,
    SORTING_EMBEDDING_CACHE_FACENET_FILENAME,
    SORTING_EMBEDDING_CACHE_INSIGHTFACE_FILENAME,
    SORTING_EMBEDDING_CACHE_MAX_ITEMS,
    SORTING_EMBEDDING_CACHE_SCHEMA_VERSION,
)
from shared.embedding_ops import safe_embedding_vector
from shared.utils import atomic_write_text, clamp_confidence, ensure_dir, safe_float, utc_now_iso


class SortedEmbeddingCacheEntry(TypedDict):
    video_path: str
    source_video_path: str
    predicted_label: str
    decision_label: str
    confidence_score: float
    embedding: List[float]
    reid_embedding: List[float]
    embedding_source: str
    updated_at: str


def sorting_embedding_cache_filename(use_insightface: bool) -> str:
    """Return the cache filename for the given engine type."""
    return (
        SORTING_EMBEDDING_CACHE_INSIGHTFACE_FILENAME
        if bool(use_insightface)
        else SORTING_EMBEDDING_CACHE_FACENET_FILENAME
    )


def embedding_model_key(use_insightface: bool) -> str:
    """Return a short key identifying the active face model."""
    return "insightface" if bool(use_insightface) else "facenet"


def sorting_embedding_cache_path(output_dir: Path, use_insightface: bool) -> Path:
    """Return the full path to the embedding cache file."""
    return output_dir / LEARNING_DIRNAME / sorting_embedding_cache_filename(use_insightface)


def normalize_cache_key(path_text: str) -> str:
    """Normalise a filesystem path for use as a cache dictionary key."""
    return os.path.normcase(os.path.normpath(str(path_text).strip()))


def load_sorting_embedding_cache(path: Path) -> Dict[str, Any]:
    """Load and validate an embedding cache from *path*, returning a safe default on error."""
    if not path.exists():
        return {"schema_version": SORTING_EMBEDDING_CACHE_SCHEMA_VERSION, "updated_at": "", "entries": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"schema_version": SORTING_EMBEDDING_CACHE_SCHEMA_VERSION, "updated_at": "", "entries": {}}
    if not isinstance(data, dict):
        return {"schema_version": SORTING_EMBEDDING_CACHE_SCHEMA_VERSION, "updated_at": "", "entries": {}}
    entries = data.get("entries", {})
    if not isinstance(entries, dict):
        entries = {}
    data["schema_version"] = SORTING_EMBEDDING_CACHE_SCHEMA_VERSION
    data["entries"] = entries
    data.setdefault("updated_at", "")
    return data


def persist_sorted_embedding_cache(
    output_dir: Path,
    results: Sequence[Dict[str, Any]],
    *,
    use_insightface: bool,
) -> Dict[str, Any]:
    cache_path = sorting_embedding_cache_path(output_dir, use_insightface=use_insightface)
    cache = load_sorting_embedding_cache(cache_path)
    entries = cache.get("entries", {})
    if not isinstance(entries, dict):
        entries = {}
        cache["entries"] = entries

    updated = 0
    now_iso = utc_now_iso()
    for item in results:
        final_destination = str(item.get("final_destination", "")).strip()
        if not final_destination:
            continue
        embedding = safe_embedding_vector(item.get("embedding"))
        if embedding is None:
            continue
        reid_embedding = safe_embedding_vector(item.get("reid_embedding")) or []

        key = normalize_cache_key(final_destination)
        payload: SortedEmbeddingCacheEntry = {
            "video_path": str(Path(final_destination).resolve()),
            "source_video_path": str(item.get("video", "")).strip(),
            "predicted_label": str(item.get("suggested_folder_name", "")).strip()
            or str(item.get("decision_label", "")).strip().lower(),
            "decision_label": str(item.get("decision_label", "")).strip().lower(),
            "confidence_score": round(clamp_confidence(safe_float(item.get("confidence_score", 0.0))), 4),
            "embedding": embedding,
            "reid_embedding": reid_embedding,
            "embedding_source": str(item.get("embedding_source", "")),
            "updated_at": now_iso,
        }
        entries[key] = payload
        updated += 1

    if len(entries) > SORTING_EMBEDDING_CACHE_MAX_ITEMS:
        ranked = sorted(
            entries.items(),
            key=lambda pair: str(pair[1].get("updated_at", "")),
            reverse=True,
        )
        entries = dict(ranked[:SORTING_EMBEDDING_CACHE_MAX_ITEMS])
        cache["entries"] = entries

    cache["schema_version"] = SORTING_EMBEDDING_CACHE_SCHEMA_VERSION
    cache["updated_at"] = now_iso
    atomic_write_text(cache_path, json.dumps(cache, indent=2, ensure_ascii=False) + "\n")

    return {
        "path": str(cache_path),
        "updated_entries": int(updated),
        "total_entries": int(len(entries)),
    }
