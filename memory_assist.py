"""Memory-assist module – apply learned identity matches and update learning from reviews."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from learning_memory import match_identity, record_feedback
from shared.utils import clamp_confidence

from explainability import set_decision


def apply_memory_assist(result: Dict[str, Any], memory: Dict[str, Any], cfg: Any) -> None:
    """Apply learned identity memory matching to a processing result.

    *cfg* must expose ``learning_enabled``, ``learning_auto_threshold``,
    ``learning_suggest_threshold``.
    """
    result["memory_match_label"] = ""
    result["memory_match_score"] = 0.0
    result["memory_applied"] = False
    result["learning_applied"] = False
    result["adaptive_threshold_used"] = round(float(cfg.learning_auto_threshold), 4)
    result["feedback_consistency_snapshot"] = {}
    result["memory_suggestion"] = str(result.get("memory_suggestion", ""))
    result["memory_match_arcface_score"] = 0.0
    result["memory_match_reid_score"] = None
    result["memory_match_rerank_mode"] = ""

    if not cfg.learning_enabled:
        return

    embedding = result.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        return

    match = match_identity(
        memory,
        embedding,
        global_auto_threshold=cfg.learning_auto_threshold,
        global_suggest_threshold=cfg.learning_suggest_threshold,
        default_same_person_threshold=float(getattr(cfg, "same_person_threshold", 0.0) or 0.0),
        reid_embedding=result.get("reid_embedding"),
        cross_video_reid=bool(getattr(cfg, "cross_video_reid", False)),
        reid_fusion_weight=float(getattr(cfg, "reid_fusion_weight", 0.35)),
        reid_min_similarity=float(getattr(cfg, "reid_min_similarity", 0.55)),
        reid_ambiguity_margin_low=float(getattr(cfg, "reid_ambiguity_margin_low", 0.08)),
        reid_ambiguity_margin_high=float(getattr(cfg, "reid_ambiguity_margin_high", 0.06)),
    )
    if not match:
        return

    label = str(match.get("label", "")).strip()
    score = float(match.get("score", 0.0))
    match_locked = bool(match.get("locked", False))
    adaptive_auto_threshold = float(match.get("adaptive_auto_threshold", cfg.learning_auto_threshold))
    adaptive_auto_threshold = max(cfg.learning_suggest_threshold, min(cfg.learning_auto_threshold, adaptive_auto_threshold))
    consistency = float(match.get("correction_consistency_score", 0.0))
    positive_count = int(match.get("positive_feedback_count", 0) or 0)
    negative_count = int(match.get("negative_feedback_count", 0) or 0)
    if not label:
        return

    result["memory_match_label"] = label
    result["memory_match_score"] = round(score, 4)
    result["memory_match_arcface_score"] = round(float(match.get("arcface_score", score)), 4)
    result["memory_match_reid_score"] = (
        None if match.get("reid_score", None) is None else round(float(match.get("reid_score", 0.0)), 4)
    )
    result["memory_match_rerank_mode"] = str(match.get("rerank_mode", ""))
    result["memory_suggestion"] = label
    result["adaptive_threshold_used"] = round(adaptive_auto_threshold, 4)
    result["feedback_consistency_snapshot"] = {
        "positive_feedback_count": positive_count,
        "negative_feedback_count": negative_count,
        "correction_consistency_score": round(consistency, 4),
    }

    base_label = str(result.get("decision_label", "")).strip().lower()
    base_conf = float(result.get("confidence_score", 0.0))

    if match_locked and score >= adaptive_auto_threshold:
        result["memory_applied"] = True
        result["learning_applied"] = True
        result["female_found"] = True
        result["suggested_folder_name"] = label
        result["suggested_cluster_id"] = None
        set_decision(
            result,
            "female_detected",
            (
                f"Applied locked identity memory match: {label} "
                f"(score={score:.3f}, adaptive_threshold={adaptive_auto_threshold:.3f})."
            ),
            max(base_conf, score),
        )
        return

    if score >= adaptive_auto_threshold:
        confidence_gap_clear = (score - base_conf) >= 0.18 or base_conf < 0.65
        if base_label == "female_detected" or confidence_gap_clear:
            result["memory_applied"] = True
            result["learning_applied"] = True
            result["female_found"] = True
            result["suggested_folder_name"] = label
            result["suggested_cluster_id"] = None
            set_decision(
                result,
                "female_detected",
                (
                    f"Applied learned identity memory match: {label} "
                    f"(score={score:.3f}, adaptive_threshold={adaptive_auto_threshold:.3f})."
                ),
                max(base_conf, score),
            )
            return

    if score >= cfg.learning_suggest_threshold:
        result["suggested_folder_name"] = label
        result["female_found"] = False
        result["suggested_cluster_id"] = None
        uncertain_confidence = max(float(cfg.learning_suggest_threshold), min(float(score), float(adaptive_auto_threshold)))
        set_decision(
            result,
            "uncertain",
            (
                f"Memory suggests {label} (score={score:.3f}) but confidence is below "
                f"adaptive auto-apply threshold ({adaptive_auto_threshold:.3f})."
            ),
            uncertain_confidence,
        )


def update_learning_from_review_item(
    updated_item: Dict[str, Any],
    memory: Dict[str, Any],
    *,
    learning_auto_threshold: float = 0.82,
    learning_suggest_threshold: float = 0.74,
) -> bool:
    action = str(updated_item.get("review_action", "")).strip().lower()
    if action not in {"approve_suggested", "move_no_female", "reassign_existing", "reassign_new"}:
        return False

    final_path = Path(str(updated_item.get("final_path", "")).strip())
    final_label = str(updated_item.get("final_label", "")).strip() or final_path.parent.name
    if not final_label:
        return False

    predicted_label = str(updated_item.get("predicted_label", ""))
    memory_match_label = str(updated_item.get("memory_match_label", "")).strip()
    suggested_folder = str(updated_item.get("suggested_folder", "")).strip()
    source_path = str(updated_item.get("source_path", ""))
    confidence = float(updated_item.get("confidence", 0.0))
    embedding = updated_item.get("embedding")
    reid_embedding = updated_item.get("reid_embedding")
    memory_match_score = float(updated_item.get("memory_match_score", 0.0))
    source_action = f"review_{action}"

    record_feedback(
        memory,
        action=action,
        source_action=source_action,
        feedback_event_type="positive",
        label=final_label,
        predicted_label=predicted_label,
        confidence=confidence,
        source_path=source_path,
        final_path=str(final_path),
        embedding=embedding,
        reid_embedding=reid_embedding,
        memory_match_label=memory_match_label,
        memory_match_score=memory_match_score,
        from_label=predicted_label,
        to_label=final_label,
        global_auto_threshold=learning_auto_threshold,
        global_suggest_threshold=learning_suggest_threshold,
    )

    negative_candidate = ""
    if action in {"reassign_existing", "reassign_new"}:
        if memory_match_label and memory_match_label.lower() != final_label.lower():
            negative_candidate = memory_match_label
        elif suggested_folder and suggested_folder.lower() != final_label.lower():
            negative_candidate = suggested_folder
    elif action == "move_no_female":
        if memory_match_label and memory_match_label.lower() != "no_female_found":
            negative_candidate = memory_match_label
        elif suggested_folder and suggested_folder.lower() != "no_female_found":
            negative_candidate = suggested_folder

    if negative_candidate:
        record_feedback(
            memory,
            action=action,
            source_action=source_action,
            feedback_event_type="negative",
            label=final_label,
            predicted_label=predicted_label,
            confidence=confidence,
            source_path=source_path,
            final_path=str(final_path),
            embedding=None,
            memory_match_label=memory_match_label,
            memory_match_score=memory_match_score,
            from_label=negative_candidate,
            to_label=final_label,
            negative_label=negative_candidate,
            global_auto_threshold=learning_auto_threshold,
            global_suggest_threshold=learning_suggest_threshold,
        )
    return True
