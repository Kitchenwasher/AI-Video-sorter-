from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from reid_fusion import rerank_similarity


MEMORY_SCHEMA_VERSION = 1
MEMORY_DIRNAME = ".learning"
MEMORY_FILENAME = "memory_v1.json"
DEFAULT_LEARNING_AUTO_THRESHOLD = 0.82
DEFAULT_LEARNING_SUGGEST_THRESHOLD = 0.74


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def default_memory_path(base_dir: Path) -> Path:
    return base_dir / MEMORY_DIRNAME / MEMORY_FILENAME


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def compute_correction_consistency_score(positive_count: int, negative_count: int) -> float:
    total = max(0, int(positive_count)) + max(0, int(negative_count))
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, float(max(0, int(positive_count))) / float(total)))


def compute_adaptive_auto_threshold(
    *,
    global_auto_threshold: float,
    global_suggest_threshold: float,
    positive_feedback_count: int,
    negative_feedback_count: int,
    correction_consistency_score: Optional[float] = None,
) -> float:
    global_auto = _clamp(float(global_auto_threshold), 0.0, 1.0)
    global_suggest = _clamp(float(global_suggest_threshold), 0.0, global_auto)
    positive = max(0, int(positive_feedback_count))
    negative = max(0, int(negative_feedback_count))

    consistency = (
        compute_correction_consistency_score(positive, negative)
        if correction_consistency_score is None
        else _clamp(float(correction_consistency_score), 0.0, 1.0)
    )

    # Conservative adaptive policy:
    # - max downward shift 0.08 from consistent positive corrections
    # - max upward shift 0.05 from negative/conflicting corrections
    positive_strength = min(1.0, positive / 20.0)
    negative_strength = min(1.0, negative / 20.0)
    bonus = min(0.08, 0.08 * positive_strength * consistency)
    penalty = min(0.05, 0.05 * negative_strength * (1.0 + max(0.0, 0.6 - consistency)))
    adaptive = global_auto - bonus + penalty
    return round(_clamp(adaptive, global_suggest, global_auto), 4)


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
            "total_positive_events": 0,
            "total_negative_events": 0,
            "total_structural_events": 0,
        },
    }


def _atomic_save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _ensure_stats(memory: Dict[str, Any]) -> Dict[str, Any]:
    stats = memory.setdefault("stats", {})
    if not isinstance(stats, dict):
        stats = {}
        memory["stats"] = stats
    stats.setdefault("total_feedback_events", 0)
    stats.setdefault("total_identity_updates", 0)
    stats.setdefault("total_no_female_events", 0)
    stats.setdefault("total_positive_events", 0)
    stats.setdefault("total_negative_events", 0)
    stats.setdefault("total_structural_events", 0)
    return stats


def _normalize_identity_record(
    identity: Dict[str, Any],
    *,
    global_auto_threshold: float,
    global_suggest_threshold: float,
) -> Dict[str, Any]:
    normalized = dict(identity)
    normalized.setdefault("label", "")
    normalized.setdefault("prototype", [])
    normalized.setdefault("sample_count", 0)
    normalized.setdefault("confidence_sum", 0.0)
    normalized.setdefault("last_used", "")
    normalized["locked"] = bool(normalized.get("locked", False))
    normalized.setdefault("locked_at", "")
    normalized.setdefault("positive_feedback_count", 0)
    normalized.setdefault("negative_feedback_count", 0)
    normalized.setdefault("correction_consistency_score", 0.0)
    normalized.setdefault("adaptive_auto_threshold", global_auto_threshold)
    normalized.setdefault("last_corrected_at", "")
    normalized.setdefault("same_person_threshold", None)
    normalized.setdefault("intra_distance_mean", None)
    normalized.setdefault("intra_distance_std", None)
    normalized.setdefault("intra_distance_pair_count", 0)
    normalized.setdefault("same_person_threshold_updated_at", "")
    normalized.setdefault("reid_prototype", [])
    normalized.setdefault("reid_sample_count", 0)
    normalized.setdefault("reid_last_used", "")
    normalized.setdefault("reid_same_person_threshold", None)

    positive = max(0, int(normalized.get("positive_feedback_count", 0) or 0))
    negative = max(0, int(normalized.get("negative_feedback_count", 0) or 0))
    normalized["positive_feedback_count"] = positive
    normalized["negative_feedback_count"] = negative
    normalized["sample_count"] = max(0, int(normalized.get("sample_count", 0) or 0))
    normalized["confidence_sum"] = float(normalized.get("confidence_sum", 0.0) or 0.0)
    normalized["last_used"] = str(normalized.get("last_used", "") or "")
    normalized["locked_at"] = str(normalized.get("locked_at", "") or "")
    normalized["last_corrected_at"] = str(normalized.get("last_corrected_at", "") or "")
    normalized["same_person_threshold_updated_at"] = str(normalized.get("same_person_threshold_updated_at", "") or "")
    normalized["intra_distance_pair_count"] = max(0, int(normalized.get("intra_distance_pair_count", 0) or 0))
    normalized["reid_sample_count"] = max(0, int(normalized.get("reid_sample_count", 0) or 0))
    normalized["reid_last_used"] = str(normalized.get("reid_last_used", "") or "")
    reid_proto = normalized.get("reid_prototype", [])
    if not isinstance(reid_proto, list):
        reid_proto = []
    normalized["reid_prototype"] = reid_proto

    same_person_threshold = normalized.get("same_person_threshold", None)
    try:
        same_person_threshold = None if same_person_threshold is None else _clamp(float(same_person_threshold), 0.0, 1.0)
    except Exception:
        same_person_threshold = None
    normalized["same_person_threshold"] = same_person_threshold
    reid_same_person_threshold = normalized.get("reid_same_person_threshold", None)
    try:
        reid_same_person_threshold = (
            None
            if reid_same_person_threshold is None
            else _clamp(float(reid_same_person_threshold), 0.0, 1.0)
        )
    except Exception:
        reid_same_person_threshold = None
    normalized["reid_same_person_threshold"] = reid_same_person_threshold

    try:
        intra_mean = None if normalized.get("intra_distance_mean", None) is None else float(normalized.get("intra_distance_mean"))
    except Exception:
        intra_mean = None
    try:
        intra_std = None if normalized.get("intra_distance_std", None) is None else float(normalized.get("intra_distance_std"))
    except Exception:
        intra_std = None
    normalized["intra_distance_mean"] = intra_mean
    normalized["intra_distance_std"] = intra_std

    consistency = compute_correction_consistency_score(positive, negative)
    adaptive = compute_adaptive_auto_threshold(
        global_auto_threshold=global_auto_threshold,
        global_suggest_threshold=global_suggest_threshold,
        positive_feedback_count=positive,
        negative_feedback_count=negative,
        correction_consistency_score=consistency,
    )
    normalized["correction_consistency_score"] = round(consistency, 4)
    normalized["adaptive_auto_threshold"] = adaptive
    return normalized


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

    identities_raw = data.get("identities")
    if not isinstance(identities_raw, list):
        data["identities"] = []
    else:
        normalized_identities: List[Dict[str, Any]] = []
        for identity in identities_raw:
            if not isinstance(identity, dict):
                continue
            normalized = _normalize_identity_record(
                identity,
                global_auto_threshold=DEFAULT_LEARNING_AUTO_THRESHOLD,
                global_suggest_threshold=DEFAULT_LEARNING_SUGGEST_THRESHOLD,
            )
            normalized_identities.append(normalized)
        data["identities"] = normalized_identities

    if not isinstance(data.get("decisions"), list):
        data["decisions"] = []
    _ensure_stats(data)
    return data


def save_memory(path: Path, memory: Dict[str, Any]) -> None:
    memory["schema_version"] = MEMORY_SCHEMA_VERSION
    memory["updated_at"] = utc_now_iso()
    _atomic_save(path, memory)


def refresh_all_identity_stats(
    memory: Dict[str, Any],
    *,
    global_auto_threshold: float = DEFAULT_LEARNING_AUTO_THRESHOLD,
    global_suggest_threshold: float = DEFAULT_LEARNING_SUGGEST_THRESHOLD,
) -> None:
    identities = memory.setdefault("identities", [])
    if not isinstance(identities, list):
        identities = []
        memory["identities"] = identities
    for idx, identity in enumerate(identities):
        if not isinstance(identity, dict):
            continue
        identities[idx] = _normalize_identity_record(
            identity,
            global_auto_threshold=global_auto_threshold,
            global_suggest_threshold=global_suggest_threshold,
        )


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


def match_identity(
    memory: Dict[str, Any],
    embedding: Sequence[float],
    *,
    global_auto_threshold: float = DEFAULT_LEARNING_AUTO_THRESHOLD,
    global_suggest_threshold: float = DEFAULT_LEARNING_SUGGEST_THRESHOLD,
    default_same_person_threshold: float = 0.0,
    reid_embedding: Optional[Sequence[float]] = None,
    cross_video_reid: bool = False,
    reid_fusion_weight: float = 0.35,
    reid_min_similarity: float = 0.55,
    reid_ambiguity_margin_low: float = 0.08,
    reid_ambiguity_margin_high: float = 0.06,
) -> Optional[Dict[str, Any]]:
    query = normalize_embedding(embedding)
    if query is None:
        return None
    query_reid = normalize_embedding(reid_embedding or [])

    refresh_all_identity_stats(
        memory,
        global_auto_threshold=global_auto_threshold,
        global_suggest_threshold=global_suggest_threshold,
    )
    identities = memory.get("identities", [])
    if not isinstance(identities, list) or not identities:
        return None

    best: Optional[Tuple[int, Dict[str, Any], float, Dict[str, Any]]] = None
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
        threshold_raw = identity.get("same_person_threshold", default_same_person_threshold)
        try:
            identity_threshold = _clamp(float(threshold_raw), 0.0, 1.0)
        except Exception:
            identity_threshold = _clamp(float(default_same_person_threshold), 0.0, 1.0)

        reid_score: Optional[float] = None
        reid_proto = normalize_embedding(identity.get("reid_prototype", []))
        if query_reid is not None and reid_proto is not None:
            reid_score = cosine_similarity(query_reid, reid_proto)

        rerank = rerank_similarity(
            arcface_similarity=score,
            threshold=identity_threshold,
            reid_similarity=reid_score,
            reid_enabled=bool(cross_video_reid),
            reid_fusion_weight=float(reid_fusion_weight),
            reid_min_similarity=float(reid_min_similarity),
            reid_ambiguity_margin_low=float(reid_ambiguity_margin_low),
            reid_ambiguity_margin_high=float(reid_ambiguity_margin_high),
        )
        if not bool(rerank.get("accepted", False)):
            continue
        fused_score = float(rerank.get("fused_score", score))
        if best is None or fused_score > best[2]:
            best = (idx, identity, fused_score, rerank)

    if best is None:
        return None

    _, identity, score, rerank = best
    return {
        "identity_index": best[0],
        "label": str(identity.get("label", "")).strip(),
        "score": round(float(score), 4),
        "arcface_score": round(float(rerank.get("arcface_score", score)), 4),
        "reid_score": (
            None
            if rerank.get("reid_score", None) is None
            else round(float(rerank.get("reid_score", 0.0)), 4)
        ),
        "fused_score": round(float(rerank.get("fused_score", score)), 4),
        "rerank_mode": str(rerank.get("mode", "")),
        "locked": bool(identity.get("locked", False)),
        "adaptive_auto_threshold": float(identity.get("adaptive_auto_threshold", global_auto_threshold)),
        "correction_consistency_score": float(identity.get("correction_consistency_score", 0.0)),
        "positive_feedback_count": int(identity.get("positive_feedback_count", 0) or 0),
        "negative_feedback_count": int(identity.get("negative_feedback_count", 0) or 0),
        "same_person_threshold": (
            _clamp(float(identity.get("same_person_threshold", default_same_person_threshold)), 0.0, 1.0)
            if identity.get("same_person_threshold", None) is not None
            else _clamp(float(default_same_person_threshold), 0.0, 1.0)
        ),
    }


def set_identity_same_person_threshold(
    memory: Dict[str, Any],
    *,
    label: str,
    same_person_threshold: float,
    intra_distance_mean: float,
    intra_distance_std: float,
    intra_distance_pair_count: int,
    global_auto_threshold: float = DEFAULT_LEARNING_AUTO_THRESHOLD,
    global_suggest_threshold: float = DEFAULT_LEARNING_SUGGEST_THRESHOLD,
) -> None:
    clean_label = str(label or "").strip()
    if not clean_label:
        return

    identity = _ensure_identity(
        memory,
        clean_label,
        global_auto_threshold=global_auto_threshold,
        global_suggest_threshold=global_suggest_threshold,
    )
    identity["same_person_threshold"] = _clamp(float(same_person_threshold), 0.0, 1.0)
    identity["intra_distance_mean"] = float(intra_distance_mean)
    identity["intra_distance_std"] = float(intra_distance_std)
    identity["intra_distance_pair_count"] = max(0, int(intra_distance_pair_count))
    identity["same_person_threshold_updated_at"] = utc_now_iso()


def _record_decision(memory: Dict[str, Any], decision: Dict[str, Any]) -> None:
    decisions = memory.setdefault("decisions", [])
    if not isinstance(decisions, list):
        decisions = []
        memory["decisions"] = decisions

    decisions.append(
        {
            "timestamp": utc_now_iso(),
            "action": str(decision.get("action", "")),
            "source_action": str(decision.get("source_action", decision.get("action", ""))),
            "feedback_event_type": str(decision.get("feedback_event_type", "")),
            "label": str(decision.get("label", "")),
            "from_label": str(decision.get("from_label", "")),
            "to_label": str(decision.get("to_label", "")),
            "predicted_label": str(decision.get("predicted_label", "")),
            "confidence": float(decision.get("confidence", 0.0)),
            "memory_match_label": str(decision.get("memory_match_label", "")),
            "memory_match_score": float(decision.get("memory_match_score", 0.0)),
            "source_path": str(decision.get("source_path", "")),
            "final_path": str(decision.get("final_path", "")),
            "embedding_present": bool(decision.get("embedding_present", False)),
        }
    )
    if len(decisions) > 7000:
        del decisions[:-7000]


def _find_identity(memory: Dict[str, Any], label: str) -> Optional[Dict[str, Any]]:
    identities = memory.setdefault("identities", [])
    if not isinstance(identities, list):
        identities = []
        memory["identities"] = identities
    key = label.strip().lower()
    if not key:
        return None
    for identity in identities:
        if not isinstance(identity, dict):
            continue
        if str(identity.get("label", "")).strip().lower() == key:
            return identity
    return None


def _ensure_identity(
    memory: Dict[str, Any],
    label: str,
    *,
    global_auto_threshold: float,
    global_suggest_threshold: float,
) -> Dict[str, Any]:
    existing = _find_identity(memory, label)
    if existing is not None:
        normalized = _normalize_identity_record(
            existing,
            global_auto_threshold=global_auto_threshold,
            global_suggest_threshold=global_suggest_threshold,
        )
        existing.clear()
        existing.update(normalized)
        return existing

    identity = _normalize_identity_record(
        {
            "label": label,
            "prototype": [],
            "sample_count": 0,
            "confidence_sum": 0.0,
            "last_used": "",
            "locked": False,
            "locked_at": "",
            "positive_feedback_count": 0,
            "negative_feedback_count": 0,
            "last_corrected_at": "",
            "reid_prototype": [],
            "reid_sample_count": 0,
            "reid_last_used": "",
            "reid_same_person_threshold": None,
        },
        global_auto_threshold=global_auto_threshold,
        global_suggest_threshold=global_suggest_threshold,
    )
    identities = memory.setdefault("identities", [])
    if not isinstance(identities, list):
        identities = []
        memory["identities"] = identities
    identities.append(identity)
    return identity


def _update_identity(
    memory: Dict[str, Any],
    label: str,
    embedding: Optional[np.ndarray],
    confidence: float,
    reid_embedding: Optional[np.ndarray] = None,
    *,
    global_auto_threshold: float,
    global_suggest_threshold: float,
) -> None:
    target = _ensure_identity(
        memory,
        label,
        global_auto_threshold=global_auto_threshold,
        global_suggest_threshold=global_suggest_threshold,
    )

    if embedding is not None:
        old_proto = normalize_embedding(target.get("prototype", []))
        sample_count = int(target.get("sample_count", 0))
        if old_proto is None or sample_count <= 0:
            target["prototype"] = embedding.astype(np.float32).tolist()
            target["sample_count"] = 1
            target["confidence_sum"] = float(confidence)
            target["last_used"] = utc_now_iso()
        else:
            merged = ((old_proto * sample_count) + embedding) / float(sample_count + 1)
            merged_norm = normalize_embedding(merged)
            if merged_norm is None:
                merged_norm = embedding

            target["prototype"] = merged_norm.astype(np.float32).tolist()
            target["sample_count"] = sample_count + 1
            target["confidence_sum"] = float(target.get("confidence_sum", 0.0)) + float(confidence)
            target["last_used"] = utc_now_iso()

    if reid_embedding is not None:
        old_reid = normalize_embedding(target.get("reid_prototype", []))
        reid_count = int(target.get("reid_sample_count", 0) or 0)
        if old_reid is None or reid_count <= 0:
            target["reid_prototype"] = reid_embedding.astype(np.float32).tolist()
            target["reid_sample_count"] = 1
            target["reid_last_used"] = utc_now_iso()
        else:
            merged_reid = ((old_reid * reid_count) + reid_embedding) / float(reid_count + 1)
            merged_reid_norm = normalize_embedding(merged_reid)
            if merged_reid_norm is None:
                merged_reid_norm = reid_embedding
            target["reid_prototype"] = merged_reid_norm.astype(np.float32).tolist()
            target["reid_sample_count"] = reid_count + 1
            target["reid_last_used"] = utc_now_iso()


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
    reid_embedding: Optional[Sequence[float]] = None,
    memory_match_label: str = "",
    memory_match_score: float = 0.0,
    feedback_event_type: str = "",
    from_label: str = "",
    to_label: str = "",
    source_action: str = "",
    is_negative: bool = False,
    negative_label: str = "",
    global_auto_threshold: float = DEFAULT_LEARNING_AUTO_THRESHOLD,
    global_suggest_threshold: float = DEFAULT_LEARNING_SUGGEST_THRESHOLD,
) -> None:
    stats = _ensure_stats(memory)
    stats["total_feedback_events"] = int(stats.get("total_feedback_events", 0)) + 1

    event_type = str(feedback_event_type or "").strip().lower()
    if not event_type:
        event_type = "negative" if is_negative else "positive"
    if event_type not in {"positive", "negative", "structural"}:
        event_type = "positive"

    final_label = str(label or "").strip()
    emb_vec = normalize_embedding(embedding or [])
    reid_vec = normalize_embedding(reid_embedding or [])

    if final_label.lower() == "no_female_found":
        stats["total_no_female_events"] = int(stats.get("total_no_female_events", 0)) + 1
    elif (emb_vec is not None or reid_vec is not None) and final_label:
        _update_identity(
            memory,
            final_label,
            emb_vec,
            confidence,
            reid_embedding=reid_vec,
            global_auto_threshold=global_auto_threshold,
            global_suggest_threshold=global_suggest_threshold,
        )
        stats["total_identity_updates"] = int(stats.get("total_identity_updates", 0)) + 1

    now = utc_now_iso()
    if event_type == "positive" and final_label and final_label.lower() != "no_female_found":
        identity = _ensure_identity(
            memory,
            final_label,
            global_auto_threshold=global_auto_threshold,
            global_suggest_threshold=global_suggest_threshold,
        )
        identity["positive_feedback_count"] = int(identity.get("positive_feedback_count", 0)) + 1
        identity["last_corrected_at"] = now
        stats["total_positive_events"] = int(stats.get("total_positive_events", 0)) + 1

    negative_target = str(negative_label or from_label or "").strip()
    if (event_type == "negative" or is_negative) and negative_target and negative_target.lower() != "no_female_found":
        identity = _ensure_identity(
            memory,
            negative_target,
            global_auto_threshold=global_auto_threshold,
            global_suggest_threshold=global_suggest_threshold,
        )
        identity["negative_feedback_count"] = int(identity.get("negative_feedback_count", 0)) + 1
        identity["last_corrected_at"] = now
        stats["total_negative_events"] = int(stats.get("total_negative_events", 0)) + 1

    if event_type == "structural":
        stats["total_structural_events"] = int(stats.get("total_structural_events", 0)) + 1

    refresh_all_identity_stats(
        memory,
        global_auto_threshold=global_auto_threshold,
        global_suggest_threshold=global_suggest_threshold,
    )

    _record_decision(
        memory,
        {
            "action": action,
            "source_action": source_action or action,
            "feedback_event_type": event_type,
            "label": final_label,
            "from_label": from_label,
            "to_label": to_label or final_label,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "memory_match_label": memory_match_label,
            "memory_match_score": memory_match_score,
            "source_path": source_path,
            "final_path": final_path,
            "embedding_present": emb_vec is not None or reid_vec is not None,
        },
    )


def record_structural_feedback(
    memory: Dict[str, Any],
    *,
    action: str,
    source_action: str,
    from_label: str,
    to_label: str,
    source_path: str,
    final_path: str,
    global_auto_threshold: float = DEFAULT_LEARNING_AUTO_THRESHOLD,
    global_suggest_threshold: float = DEFAULT_LEARNING_SUGGEST_THRESHOLD,
) -> None:
    record_feedback(
        memory,
        action=action,
        label=to_label,
        predicted_label="",
        confidence=0.0,
        source_path=source_path,
        final_path=final_path,
        embedding=None,
        memory_match_label="",
        memory_match_score=0.0,
        feedback_event_type="structural",
        from_label=from_label,
        to_label=to_label,
        source_action=source_action,
        global_auto_threshold=global_auto_threshold,
        global_suggest_threshold=global_suggest_threshold,
    )


def _correction_trend(positive: int, negative: int) -> str:
    if positive <= 0 and negative <= 0:
        return "no_data"
    if positive >= (negative * 2) and positive >= 2:
        return "improving"
    if negative > positive:
        return "conflicting"
    return "mixed"


def build_learning_summary(
    memory: Dict[str, Any],
    *,
    global_auto_threshold: float = DEFAULT_LEARNING_AUTO_THRESHOLD,
    global_suggest_threshold: float = DEFAULT_LEARNING_SUGGEST_THRESHOLD,
    limit: int = 200,
) -> Dict[str, Any]:
    refresh_all_identity_stats(
        memory,
        global_auto_threshold=global_auto_threshold,
        global_suggest_threshold=global_suggest_threshold,
    )
    stats = _ensure_stats(memory)
    identities = memory.get("identities", [])
    if not isinstance(identities, list):
        identities = []

    rows: List[Dict[str, Any]] = []
    for identity in identities:
        if not isinstance(identity, dict):
            continue
        label = str(identity.get("label", "")).strip()
        if not label:
            continue
        positive = int(identity.get("positive_feedback_count", 0) or 0)
        negative = int(identity.get("negative_feedback_count", 0) or 0)
        rows.append(
            {
                "label": label,
                "locked": bool(identity.get("locked", False)),
                "sample_count": int(identity.get("sample_count", 0) or 0),
                "reid_sample_count": int(identity.get("reid_sample_count", 0) or 0),
                "positive_feedback_count": positive,
                "negative_feedback_count": negative,
                "correction_consistency_score": round(float(identity.get("correction_consistency_score", 0.0)), 4),
                "adaptive_auto_threshold": round(float(identity.get("adaptive_auto_threshold", global_auto_threshold)), 4),
                "same_person_threshold": (
                    None
                    if identity.get("same_person_threshold", None) is None
                    else round(float(identity.get("same_person_threshold", 0.0)), 4)
                ),
                "reid_same_person_threshold": (
                    None
                    if identity.get("reid_same_person_threshold", None) is None
                    else round(float(identity.get("reid_same_person_threshold", 0.0)), 4)
                ),
                "intra_distance_mean": (
                    None
                    if identity.get("intra_distance_mean", None) is None
                    else round(float(identity.get("intra_distance_mean", 0.0)), 4)
                ),
                "intra_distance_std": (
                    None
                    if identity.get("intra_distance_std", None) is None
                    else round(float(identity.get("intra_distance_std", 0.0)), 4)
                ),
                "intra_distance_pair_count": int(identity.get("intra_distance_pair_count", 0) or 0),
                "same_person_threshold_updated_at": str(identity.get("same_person_threshold_updated_at", "") or ""),
                "last_corrected_at": str(identity.get("last_corrected_at", "") or ""),
                "last_used": str(identity.get("last_used", "") or ""),
                "recent_trend": _correction_trend(positive, negative),
            }
        )

    rows.sort(
        key=lambda row: (
            row.get("last_corrected_at", ""),
            row.get("last_used", ""),
            row.get("label", "").lower(),
        ),
        reverse=True,
    )
    if limit > 0:
        rows = rows[: max(1, int(limit))]

    return {
        "ok": True,
        "count": len(rows),
        "global_auto_threshold": round(float(global_auto_threshold), 4),
        "global_suggest_threshold": round(float(global_suggest_threshold), 4),
        "items": rows,
        "stats": {
            "total_feedback_events": int(stats.get("total_feedback_events", 0)),
            "total_positive_events": int(stats.get("total_positive_events", 0)),
            "total_negative_events": int(stats.get("total_negative_events", 0)),
            "total_structural_events": int(stats.get("total_structural_events", 0)),
            "total_identity_updates": int(stats.get("total_identity_updates", 0)),
            "total_no_female_events": int(stats.get("total_no_female_events", 0)),
        },
    }
