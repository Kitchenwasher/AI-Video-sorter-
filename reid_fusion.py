"""Shared ArcFace + Re-ID fusion policy helpers."""

from __future__ import annotations

from typing import Any, Dict, Optional


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def rerank_similarity(
    *,
    arcface_similarity: float,
    threshold: float,
    reid_similarity: Optional[float],
    reid_enabled: bool,
    reid_fusion_weight: float,
    reid_min_similarity: float,
    reid_ambiguity_margin_low: float,
    reid_ambiguity_margin_high: float,
) -> Dict[str, Any]:
    """Apply adaptive ambiguous-region reranking.

    Policy:
    - arcface >= threshold + margin_high: direct accept
    - arcface < threshold - margin_low: direct reject
    - otherwise ambiguous:
      - if Re-ID unavailable, fall back to arcface threshold
      - if Re-ID available, fused = (1-w)*arc + w*reid and accept when:
        fused >= threshold and reid >= reid_min_similarity
    """
    arc = float(arcface_similarity)
    thr = float(threshold)
    weight = _clamp(float(reid_fusion_weight), 0.0, 1.0)
    margin_low = max(0.0, float(reid_ambiguity_margin_low))
    margin_high = max(0.0, float(reid_ambiguity_margin_high))
    min_reid = _clamp(float(reid_min_similarity), 0.0, 1.0)

    if arc >= (thr + margin_high):
        return {
            "accepted": True,
            "fused_score": float(arc),
            "arcface_score": float(arc),
            "reid_score": None,
            "reid_used": False,
            "mode": "direct_accept",
            "ambiguous": False,
        }

    if arc < (thr - margin_low):
        return {
            "accepted": False,
            "fused_score": float(arc),
            "arcface_score": float(arc),
            "reid_score": None,
            "reid_used": False,
            "mode": "direct_reject",
            "ambiguous": False,
        }

    # Ambiguous region around threshold.
    if not reid_enabled or reid_similarity is None:
        accepted = arc >= thr
        return {
            "accepted": bool(accepted),
            "fused_score": float(arc),
            "arcface_score": float(arc),
            "reid_score": None,
            "reid_used": False,
            "mode": "ambiguous_arcface_fallback",
            "ambiguous": True,
        }

    reid = float(reid_similarity)
    fused = ((1.0 - weight) * arc) + (weight * reid)
    accepted = fused >= thr and reid >= min_reid
    return {
        "accepted": bool(accepted),
        "fused_score": float(fused),
        "arcface_score": float(arc),
        "reid_score": float(reid),
        "reid_used": True,
        "mode": "ambiguous_fused",
        "ambiguous": True,
    }

