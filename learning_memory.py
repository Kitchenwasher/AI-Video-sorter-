from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


MEMORY_SCHEMA_VERSION = 1
MEMORY_DIRNAME = ".learning"
MEMORY_FILENAME = "memory_v1.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_memory_path(base_dir: Path) -> Path:
    return base_dir / MEMORY_DIRNAME / MEMORY_FILENAME


def _empty_memory() -> Dict[str, Any]:
    now = utc_now_iso()
    return {
        "schema_version": MEMORY_SCHEMA_VERSION,
        "updated_at": now,
        "identities": [],
        "decisions": [],
        "stats": {
            "total_feedback_events": 0,
            "total_identity_updates": 0,
            "total_no_female_events": 0,
        },
    }


def _atomic_save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def load_memory(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return _empty_memory()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _empty_memory()
    if not isinstance(data, dict):
        return _empty_memory()

    data.setdefault("schema_version", MEMORY_SCHEMA_VERSION)
    data.setdefault("updated_at", utc_now_iso())
    if not isinstance(data.get("identities"), list):
        data["identities"] = []
    if not isinstance(data.get("decisions"), list):
        data["decisions"] = []
    if not isinstance(data.get("stats"), dict):
        data["stats"] = {}
    stats = data["stats"]
    stats.setdefault("total_feedback_events", 0)
    stats.setdefault("total_identity_updates", 0)
    stats.setdefault("total_no_female_events", 0)
    return data


def save_memory(path: Path, memory: Dict[str, Any]) -> None:
    memory["schema_version"] = MEMORY_SCHEMA_VERSION
    memory["updated_at"] = utc_now_iso()
    _atomic_save(path, memory)


def normalize_embedding(embedding: Sequence[float]) -> Optional[np.ndarray]:
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.ndim != 1 or arr.size == 0 or not np.isfinite(arr).all():
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return None
    return (arr / norm).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def match_identity(memory: Dict[str, Any], embedding: Sequence[float]) -> Optional[Dict[str, Any]]:
    query = normalize_embedding(embedding)
    if query is None:
        return None

    identities = memory.get("identities", [])
    if not isinstance(identities, list) or not identities:
        return None

    best: Optional[Tuple[int, str, float]] = None
    for idx, identity in enumerate(identities):
        if not isinstance(identity, dict):
            continue
        label = str(identity.get("label", "")).strip()
        proto = identity.get("prototype")
        if not label or not isinstance(proto, list):
            continue
        prototype_vec = normalize_embedding(proto)
        if prototype_vec is None:
            continue
        score = cosine_similarity(query, prototype_vec)
        if best is None or score > best[2]:
            best = (idx, label, score)

    if best is None:
        return None

    return {
        "identity_index": best[0],
        "label": best[1],
        "score": round(float(best[2]), 4),
    }


def _record_decision(memory: Dict[str, Any], decision: Dict[str, Any]) -> None:
    decisions = memory.setdefault("decisions", [])
    if not isinstance(decisions, list):
        decisions = []
        memory["decisions"] = decisions

    decisions.append(
        {
            "timestamp": utc_now_iso(),
            "action": str(decision.get("action", "")),
            "label": str(decision.get("label", "")),
            "predicted_label": str(decision.get("predicted_label", "")),
            "confidence": float(decision.get("confidence", 0.0)),
            "memory_match_label": str(decision.get("memory_match_label", "")),
            "memory_match_score": float(decision.get("memory_match_score", 0.0)),
            "source_path": str(decision.get("source_path", "")),
            "final_path": str(decision.get("final_path", "")),
            "embedding_present": bool(decision.get("embedding_present", False)),
        }
    )
    if len(decisions) > 5000:
        del decisions[:-5000]


def _update_identity(memory: Dict[str, Any], label: str, embedding: np.ndarray, confidence: float) -> None:
    identities = memory.setdefault("identities", [])
    if not isinstance(identities, list):
        identities = []
        memory["identities"] = identities

    target: Optional[Dict[str, Any]] = None
    for identity in identities:
        if isinstance(identity, dict) and str(identity.get("label", "")) == label:
            target = identity
            break

    if target is None:
        identities.append(
            {
                "label": label,
                "prototype": embedding.astype(np.float32).tolist(),
                "sample_count": 1,
                "confidence_sum": float(confidence),
                "last_used": utc_now_iso(),
            }
        )
        return

    old_proto = normalize_embedding(target.get("prototype", []))
    sample_count = int(target.get("sample_count", 0))
    if old_proto is None or sample_count <= 0:
        target["prototype"] = embedding.astype(np.float32).tolist()
        target["sample_count"] = 1
        target["confidence_sum"] = float(confidence)
        target["last_used"] = utc_now_iso()
        return

    merged = ((old_proto * sample_count) + embedding) / float(sample_count + 1)
    merged_norm = normalize_embedding(merged)
    if merged_norm is None:
        merged_norm = embedding

    target["prototype"] = merged_norm.astype(np.float32).tolist()
    target["sample_count"] = sample_count + 1
    target["confidence_sum"] = float(target.get("confidence_sum", 0.0)) + float(confidence)
    target["last_used"] = utc_now_iso()


def record_feedback(
    memory: Dict[str, Any],
    *,
    action: str,
    label: str,
    predicted_label: str,
    confidence: float,
    source_path: str,
    final_path: str,
    embedding: Optional[Sequence[float]] = None,
    memory_match_label: str = "",
    memory_match_score: float = 0.0,
) -> None:
    stats = memory.setdefault("stats", {})
    if not isinstance(stats, dict):
        stats = {}
        memory["stats"] = stats
    stats["total_feedback_events"] = int(stats.get("total_feedback_events", 0)) + 1

    emb_vec = normalize_embedding(embedding or [])
    if label.lower() == "no_female_found":
        stats["total_no_female_events"] = int(stats.get("total_no_female_events", 0)) + 1
    elif emb_vec is not None:
        _update_identity(memory, label, emb_vec, confidence)
        stats["total_identity_updates"] = int(stats.get("total_identity_updates", 0)) + 1

    _record_decision(
        memory,
        {
            "action": action,
            "label": label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "memory_match_label": memory_match_label,
            "memory_match_score": memory_match_score,
            "source_path": source_path,
            "final_path": final_path,
            "embedding_present": emb_vec is not None,
        },
    )
