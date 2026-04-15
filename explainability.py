"""Explainability module – result records, reason tags, JSON payloads, and memory assist."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from shared.constants import REASON_TAG_PRIORITY, RESULT_JSON_PREFIX
from shared.utils import clamp_confidence, safe_float, safe_int

# Re-export what the main module originally defined here.
# Keeps backward compat with tests importing from the main module.


def empty_reason_metrics() -> Dict[str, Any]:
    return {
        "total_faces_evaluated": 0,
        "low_face_area_rejections": 0,
        "stable_embeddings": 0,
        "gender_votes": 0,
        "female_score": 0.0,
        "male_score": 0.0,
        "female_seed_hits": 0,
        "best_seed_confidence": 0.0,
    }


def set_decision(result: Dict[str, Any], label: str, reason: str, confidence: float) -> None:
    result["decision_label"] = label
    result["decision_reason"] = reason
    result["confidence_score"] = round(clamp_confidence(confidence), 3)


def new_result_record(video_path: str, device_label: str) -> Dict[str, Any]:
    return {
        "video": video_path,
        "female_found": False,
        "embedding": None,
        "embedding_source": "",
        "error": None,
        "stopped": False,
        "samples_used": 0,
        "device": device_label,
        "decision_label": "unknown",
        "decision_reason": "",
        "confidence_score": 0.0,
        "suggested_cluster_id": None,
        "suggested_folder_name": "",
        "memory_match_label": "",
        "memory_match_score": 0.0,
        "memory_applied": False,
        "learning_applied": False,
        "adaptive_threshold_used": 0.0,
        "feedback_consistency_snapshot": {},
        "memory_suggestion": "",
        "reason_summary": "",
        "reason_tags": [],
        "reason_metrics": empty_reason_metrics(),
    }


def _normalize_reason_metrics(metrics: Any) -> Dict[str, Any]:
    if not isinstance(metrics, dict):
        metrics = {}
    normalized = empty_reason_metrics()
    normalized["total_faces_evaluated"] = max(0, safe_int(metrics.get("total_faces_evaluated", 0)))
    normalized["low_face_area_rejections"] = max(0, safe_int(metrics.get("low_face_area_rejections", 0)))
    normalized["stable_embeddings"] = max(0, safe_int(metrics.get("stable_embeddings", 0)))
    normalized["gender_votes"] = max(0, safe_int(metrics.get("gender_votes", 0)))
    normalized["female_seed_hits"] = max(0, safe_int(metrics.get("female_seed_hits", 0)))
    normalized["female_score"] = round(max(0.0, safe_float(metrics.get("female_score", 0.0))), 4)
    normalized["male_score"] = round(max(0.0, safe_float(metrics.get("male_score", 0.0))), 4)
    normalized["best_seed_confidence"] = round(
        clamp_confidence(safe_float(metrics.get("best_seed_confidence", 0.0))),
        4,
    )
    return normalized


def _fallback_reason_summary(decision_label: str, decision_reason: str) -> str:
    decision_reason = decision_reason.strip()
    if decision_reason:
        return decision_reason
    label = decision_label.strip().lower()
    if label == "female_detected":
        return "Consistent female detection."
    if label == "no_female":
        return "No female evidence detected."
    if label == "uncertain":
        return "Uncertain due to weak or conflicting evidence."
    if label == "error":
        return "Processing error."
    if label == "stopped":
        return "Stopped before completion."
    return "No explanation available."


def apply_explainability_metadata(result: Dict[str, Any], cfg: Any) -> None:
    """Attach reason tags and a human-readable summary to a result dict.

    *cfg* must expose ``min_stable_embeddings`` and ``min_stabilization_gender_votes``.
    """
    metrics = _normalize_reason_metrics(result.get("reason_metrics"))
    result["reason_metrics"] = metrics

    decision_label = str(result.get("decision_label", "")).strip().lower()
    decision_reason = str(result.get("decision_reason", ""))
    reason_lower = decision_reason.lower()

    tags: List[str] = []

    total_faces = int(metrics.get("total_faces_evaluated", 0))
    low_area_rejections = int(metrics.get("low_face_area_rejections", 0))
    stable_embeddings = int(metrics.get("stable_embeddings", 0))
    gender_votes = int(metrics.get("gender_votes", 0))
    female_score = float(metrics.get("female_score", 0.0))
    male_score = float(metrics.get("male_score", 0.0))
    seed_hits = int(metrics.get("female_seed_hits", 0))

    if total_faces > 0 and low_area_rejections > 0 and (low_area_rejections / max(1, total_faces)) >= 0.5:
        tags.append("low_face_area")

    min_stable = getattr(cfg, "min_stable_embeddings", 3)
    min_gender_votes = getattr(cfg, "min_stabilization_gender_votes", 3)

    if (
        stable_embeddings > 0 and stable_embeddings < min_stable
    ) or ("stable embeddings" in reason_lower) or ("identity could not be stabilized" in reason_lower):
        tags.append("few_stable_embeddings")

    if (gender_votes > 0 and gender_votes < min_gender_votes) or (
        "only" in reason_lower and "gender votes" in reason_lower
    ):
        tags.append("few_votes")

    if (female_score > 0 and male_score > 0 and female_score <= male_score) or ("conflicting gender votes" in reason_lower):
        tags.append("gender_disagreement")

    if seed_hits > 0 and decision_label in {"uncertain", "no_female"} and "none verified" in reason_lower:
        tags.append("unverified_female_candidates")

    if bool(result.get("memory_applied", False)):
        tags.append("memory_match_applied")
    elif str(result.get("memory_match_label", "")).strip() and decision_label == "uncertain":
        tags.append("memory_match_suggested")

    # Keep deterministic order even when multiple conditions match.
    tag_set = set(tags)
    ordered_tags = [tag for tag in REASON_TAG_PRIORITY if tag in tag_set]
    result["reason_tags"] = ordered_tags

    summary_by_tag = {
        "memory_match_applied": "High-confidence memory match auto-applied.",
        "gender_disagreement": "Gender evidence disagreed during stabilization.",
        "few_stable_embeddings": "Too few stable identity samples were collected.",
        "few_votes": "Too few gender votes were collected.",
        "low_face_area": "Most candidate faces were too small for reliable evidence.",
        "unverified_female_candidates": "Female-like candidates appeared but were not verified.",
        "memory_match_suggested": "Memory suggests a match, but confidence is below auto-apply threshold.",
    }
    if ordered_tags:
        result["reason_summary"] = summary_by_tag.get(ordered_tags[0], _fallback_reason_summary(decision_label, decision_reason))
    else:
        result["reason_summary"] = _fallback_reason_summary(decision_label, decision_reason)


def build_result_json_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    video_path = str(result.get("video", ""))
    reason_tags = result.get("reason_tags", [])
    if not isinstance(reason_tags, list):
        reason_tags = []
    embedding_value = result.get("embedding")
    safe_embedding = None
    if isinstance(embedding_value, list):
        try:
            safe_embedding = [float(v) for v in embedding_value]
        except Exception:
            safe_embedding = None
    reid_embedding_value = result.get("reid_embedding")
    safe_reid_embedding = None
    if isinstance(reid_embedding_value, list):
        try:
            safe_reid_embedding = [float(v) for v in reid_embedding_value]
        except Exception:
            safe_reid_embedding = None
    payload = {
        "video": video_path,
        "video_name": Path(video_path).name if video_path else "",
        "decision_label": str(result.get("decision_label", "")),
        "confidence_score": round(clamp_confidence(safe_float(result.get("confidence_score", 0.0))), 3),
        "reason_summary": str(result.get("reason_summary", "")),
        "reason_tags": [str(tag) for tag in reason_tags],
        "reason_metrics": _normalize_reason_metrics(result.get("reason_metrics")),
        "decision_reason": str(result.get("decision_reason", "")),
        "memory_match_label": str(result.get("memory_match_label", "")),
        "memory_match_score": round(safe_float(result.get("memory_match_score", 0.0)), 4),
        "memory_match_arcface_score": round(safe_float(result.get("memory_match_arcface_score", 0.0)), 4),
        "memory_match_reid_score": (
            None
            if result.get("memory_match_reid_score", None) is None
            else round(safe_float(result.get("memory_match_reid_score", 0.0)), 4)
        ),
        "memory_match_rerank_mode": str(result.get("memory_match_rerank_mode", "")),
        "memory_applied": bool(result.get("memory_applied", False)),
        "learning_applied": bool(result.get("learning_applied", result.get("memory_applied", False))),
        "adaptive_threshold_used": round(safe_float(result.get("adaptive_threshold_used", 0.0)), 4),
        "feedback_consistency_snapshot": result.get("feedback_consistency_snapshot", {}),
        "suggested_folder_name": str(result.get("suggested_folder_name", "")),
        "suggested_cluster_id": result.get("suggested_cluster_id"),
        "samples_used": max(0, safe_int(result.get("samples_used", 0))),
        "embedding": safe_embedding,
        "embedding_source": str(result.get("embedding_source", "")),
        "reid_enabled": bool(result.get("reid_enabled", False)),
        "reid_embedding_present": bool(safe_reid_embedding),
        "reid_backend": str(result.get("reid_backend", "")),
        "reid_model_tier": str(result.get("reid_model_tier", "")),
        "reid_embedding": safe_reid_embedding,
        "stopped": bool(result.get("stopped", False)),
        "error": str(result.get("error") or ""),
    }
    return payload


def emit_result_json(result: Dict[str, Any]) -> None:
    payload = build_result_json_payload(result)
    print(f"{RESULT_JSON_PREFIX}{json.dumps(payload, separators=(',', ':'), ensure_ascii=True)}", flush=True)


def emit_progress(done: int, total: int, female: int, no_female: int, errors: int) -> None:
    print(
        f"[PROGRESS] done={done} total={total} female={female} no_female={no_female} errors={errors}",
        flush=True,
    )
