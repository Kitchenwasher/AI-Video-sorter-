"""InsightFace engine wrapper (RetinaFace + ArcFace) for embedding and matching.

This module is intentionally self-contained so existing pipeline code can switch
between legacy FaceNet and InsightFace without changing output contracts.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from insightface.app import FaceAnalysis
except Exception as exc:  # pragma: no cover - import failure path depends on runtime env.
    FaceAnalysis = None  # type: ignore[assignment]
    _IMPORT_ERROR: Optional[Exception] = exc
else:
    _IMPORT_ERROR = None

try:
    import onnxruntime as ort
except Exception:
    ort = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

_DET_SIZE: Tuple[int, int] = (640, 640)
_MAX_DETECTION_SIDE = 1280

_APP_LOCK = threading.Lock()
_APP: Optional[Any] = None
_APP_CONTEXT_LABEL = "uninitialized"
_APP_PROVIDER_LABEL = "unknown"


def _normalize_embedding(raw_embedding: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    if raw_embedding is None:
        return None
    arr = np.asarray(raw_embedding, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return None
    return (arr / norm).astype(np.float32)


def _resize_for_detection(image_bgr: np.ndarray, max_side: int = _MAX_DETECTION_SIDE) -> Tuple[np.ndarray, float]:
    """Resize very large frames for faster detection; return resized frame + reverse scale."""
    height, width = image_bgr.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image_bgr, 1.0

    scale = float(max_side) / float(longest_side)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def _preferred_providers() -> List[str]:
    """Pick the best ONNX Runtime provider order for this machine."""
    if ort is None:
        return ["CPUExecutionProvider"]
    available = set(ort.get_available_providers() or [])
    for gpu_provider in ("DmlExecutionProvider", "ROCMExecutionProvider", "CUDAExecutionProvider"):
        if gpu_provider in available:
            return [gpu_provider, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _provider_label(providers: Sequence[str]) -> str:
    first = str(providers[0]) if providers else "CPUExecutionProvider"
    if first == "DmlExecutionProvider":
        return "GPU via DirectML (InsightFace)"
    if first == "ROCMExecutionProvider":
        return "GPU via ROCm ONNX Runtime (InsightFace)"
    if first == "CUDAExecutionProvider":
        return "GPU via CUDA ONNX Runtime (InsightFace)"
    return "CPU fallback (InsightFace)"


def initialize_insightface() -> str:
    """Initialize FaceAnalysis once with GPU-first policy and CPU fallback.

    Returns a short backend label used for logs/runtime metadata.
    """
    global _APP, _APP_CONTEXT_LABEL, _APP_PROVIDER_LABEL

    if _APP is not None:
        return _APP_CONTEXT_LABEL

    if FaceAnalysis is None:
        raise RuntimeError(f"insightface import failed: {_IMPORT_ERROR}")

    with _APP_LOCK:
        if _APP is not None:
            return _APP_CONTEXT_LABEL

        providers = _preferred_providers()
        app = FaceAnalysis(name="buffalo_l", providers=providers)
        _APP_PROVIDER_LABEL = ",".join(providers)
        ctx_id = 0 if providers and providers[0] != "CPUExecutionProvider" else -1

        try:
            app.prepare(ctx_id=ctx_id, det_size=_DET_SIZE)
            _APP = app
            _APP_CONTEXT_LABEL = _provider_label(providers)
            return _APP_CONTEXT_LABEL
        except Exception as gpu_exc:
            logger.warning("InsightFace init with providers=%s failed, falling back to CPU: %s", providers, gpu_exc)

        try:
            app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=_DET_SIZE)
            _APP = app
            _APP_CONTEXT_LABEL = "CPU fallback (InsightFace)"
            _APP_PROVIDER_LABEL = "CPUExecutionProvider"
            return _APP_CONTEXT_LABEL
        except Exception as cpu_exc:
            raise RuntimeError(f"InsightFace init failed on GPU and CPU: {cpu_exc}") from cpu_exc


def _clamp_bbox(bbox: Sequence[float], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def get_faces(image: np.ndarray) -> List[Dict[str, Any]]:
    """Return InsightFace detections formatted for the existing pipeline."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return []

    initialize_insightface()
    assert _APP is not None

    src_h, src_w = image.shape[:2]
    resized, scale = _resize_for_detection(image)

    faces = _APP.get(resized)
    if not faces:
        return []

    detections: List[Dict[str, Any]] = []
    inv_scale = 1.0 / scale
    for face in faces:
        bbox_raw = getattr(face, "bbox", None)
        if bbox_raw is None:
            continue

        bbox_scaled = np.asarray(bbox_raw, dtype=np.float32) * inv_scale
        bbox = _clamp_bbox(bbox_scaled.tolist(), src_w, src_h)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        embedding_raw = getattr(face, "normed_embedding", None)
        if embedding_raw is None:
            embedding_raw = getattr(face, "embedding", None)
        embedding = _normalize_embedding(embedding_raw)

        detections.append(
            {
                "bbox": bbox,
                "prob": float(getattr(face, "det_score", 1.0)),
                "area": float((x2 - x1) * (y2 - y1)),
                "embedding": embedding,
            }
        )

    detections.sort(key=lambda item: (item["area"] * float(item["prob"])), reverse=True)
    return detections


def get_face_embedding(image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """Input: BGR image (numpy array)
    Output: embedding (numpy array), bbox
    """
    faces = get_faces(image)
    if not faces:
        return None, None

    first_face = faces[0]
    embedding = first_face.get("embedding")
    bbox = first_face.get("bbox")

    if embedding is None:
        return None, bbox

    return np.asarray(embedding, dtype=np.float32), bbox


def compare_embeddings(emb1: Sequence[float], emb2: Sequence[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    a = _normalize_embedding(emb1)
    b = _normalize_embedding(emb2)
    if a is None or b is None:
        return -1.0
    denom = (float(np.linalg.norm(a)) * float(np.linalg.norm(b))) + 1e-8
    return float(np.dot(a, b) / denom)


def reset_engine_for_tests() -> None:
    """Testing helper: clear cached model state so init can re-run."""
    global _APP, _APP_CONTEXT_LABEL, _APP_PROVIDER_LABEL
    with _APP_LOCK:
        _APP = None
        _APP_CONTEXT_LABEL = "uninitialized"
        _APP_PROVIDER_LABEL = "unknown"


def provider_label() -> str:
    return _APP_PROVIDER_LABEL
