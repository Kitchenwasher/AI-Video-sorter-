"""Canonical embedding math – cosine similarity, normalisation, robust averaging."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def normalize_embedding(embedding: Sequence[float]) -> Optional[np.ndarray]:
    """L2-normalise an embedding vector.  Returns ``None`` on degenerate input."""
    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    denom = float(np.linalg.norm(arr))
    if denom <= 1e-8:
        return None
    return (arr / denom).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    denom = (float(np.linalg.norm(a)) * float(np.linalg.norm(b))) + 1e-8
    return float(np.dot(a.ravel(), b.ravel()) / denom)


def robust_average_embeddings(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    """Compute a robust average that discards the worst 20 % of samples."""
    matrix = np.vstack(embeddings).astype(np.float32)
    center = np.mean(matrix, axis=0)
    norm = float(np.linalg.norm(center))
    if norm > 1e-8:
        center /= norm
    sims = matrix @ center
    keep_count = max(1, int(np.ceil(len(matrix) * 0.8)))
    keep_indices = np.argsort(sims)[-keep_count:]
    stable = np.mean(matrix[keep_indices], axis=0).astype(np.float32)
    norm_stable = float(np.linalg.norm(stable))
    if norm_stable > 1e-8:
        stable /= norm_stable
    return np.asarray(stable, dtype=np.float32)


def safe_embedding_vector(raw_embedding: object) -> Optional[List[float]]:
    """Validate and convert a raw embedding to a clean ``list[float]``."""
    if not isinstance(raw_embedding, list) or not raw_embedding:
        return None
    vector: List[float] = []
    try:
        for item in raw_embedding:
            value = float(item)
            if not np.isfinite(value):
                return None
            vector.append(value)
    except Exception:
        return None
    if not vector:
        return None
    return vector
