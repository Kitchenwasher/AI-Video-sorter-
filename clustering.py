"""Clustering module – DBSCAN-based identity clustering with merge logic."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.cluster import DBSCAN  # type: ignore[import-untyped]

from reid_fusion import rerank_similarity
from shared.embedding_ops import cosine_similarity, robust_average_embeddings


def cluster_embeddings(
    video_results: Sequence[Dict[str, Any]],
    eps: float,
    min_samples: int,
    cluster_merge_threshold: float = 0.78,
    cross_video_reid: bool = False,
    reid_fusion_weight: float = 0.35,
    reid_min_similarity: float = 0.55,
    reid_ambiguity_margin_low: float = 0.08,
    reid_ambiguity_margin_high: float = 0.06,
) -> Dict[str, int]:
    """Cluster video results by embedding similarity using DBSCAN.

    Returns a mapping of ``video_path -> cluster_id`` (1-indexed).
    """
    vectors: List[np.ndarray] = []
    reid_vectors: List[Optional[np.ndarray]] = []
    videos: List[str] = []

    for item in video_results:
        emb = item.get("embedding")
        if emb is None:
            continue
        vec = np.asarray(emb, dtype=np.float32)
        if vec.ndim != 1 or not np.isfinite(vec).all():
            continue
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm
        vectors.append(vec)
        reid_raw = item.get("reid_embedding")
        reid_vec: Optional[np.ndarray] = None
        if isinstance(reid_raw, list) and reid_raw:
            candidate = np.asarray(reid_raw, dtype=np.float32).reshape(-1)
            if candidate.ndim == 1 and candidate.size > 0 and np.isfinite(candidate).all():
                reid_norm = float(np.linalg.norm(candidate))
                if reid_norm > 1e-8:
                    reid_vec = (candidate / reid_norm).astype(np.float32)
        reid_vectors.append(reid_vec)
        videos.append(item["video"])

    if not vectors:
        return {}

    matrix = np.vstack(vectors)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = dbscan.fit_predict(matrix)

    cluster_vectors: Dict[int, List[np.ndarray]] = {}
    cluster_reid_vectors: Dict[int, List[np.ndarray]] = {}
    for vec, reid_vec, label in zip(vectors, reid_vectors, labels):
        cluster_vectors.setdefault(int(label), []).append(vec)
        if reid_vec is not None:
            cluster_reid_vectors.setdefault(int(label), []).append(reid_vec)

    cluster_centroids: Dict[int, np.ndarray] = {}
    for label, vecs in cluster_vectors.items():
        centroid = robust_average_embeddings(vecs)
        cluster_centroids[label] = centroid
    cluster_reid_centroids: Dict[int, np.ndarray] = {}
    for label, vecs in cluster_reid_vectors.items():
        centroid = robust_average_embeddings(vecs)
        cluster_reid_centroids[label] = centroid

    merged_labels: Dict[int, int] = {}
    next_group = 0
    labels_sorted = sorted(cluster_centroids.keys())
    for label in labels_sorted:
        if label in merged_labels:
            continue
        merged_labels[label] = next_group
        base = cluster_centroids[label]
        for other in labels_sorted:
            if other in merged_labels:
                continue
            arcface_sim = cosine_similarity(base, cluster_centroids[other])
            reid_sim = None
            base_reid = cluster_reid_centroids.get(label)
            other_reid = cluster_reid_centroids.get(other)
            if base_reid is not None and other_reid is not None:
                reid_sim = cosine_similarity(base_reid, other_reid)
            rerank = rerank_similarity(
                arcface_similarity=arcface_sim,
                threshold=cluster_merge_threshold,
                reid_similarity=reid_sim,
                reid_enabled=bool(cross_video_reid),
                reid_fusion_weight=float(reid_fusion_weight),
                reid_min_similarity=float(reid_min_similarity),
                reid_ambiguity_margin_low=float(reid_ambiguity_margin_low),
                reid_ambiguity_margin_high=float(reid_ambiguity_margin_high),
            )
            if bool(rerank.get("accepted", False)):
                merged_labels[other] = next_group
        next_group += 1

    ordered_labels = sorted(set(merged_labels[int(label)] for label in labels))
    folder_ids = {old: idx + 1 for idx, old in enumerate(ordered_labels)}
    return {video: folder_ids[merged_labels[int(label)]] for video, label in zip(videos, labels)}
