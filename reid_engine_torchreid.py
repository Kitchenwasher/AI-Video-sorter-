"""Torchreid wrapper with lazy singleton init and safe fallbacks."""

from __future__ import annotations

import logging
import threading
from typing import Optional, Sequence

import cv2
import numpy as np
import torch

try:
    from torchreid.utils import FeatureExtractor
except Exception as exc:  # pragma: no cover - import path depends on runtime env
    FeatureExtractor = None  # type: ignore[assignment]
    _IMPORT_ERROR: Optional[Exception] = exc
else:
    _IMPORT_ERROR = None


logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_EXTRACTOR: Optional[object] = None
_INIT_FAILED = False
_BACKEND_LABEL = "disabled"
_ACTIVE_MODEL_TIER = "balanced"
_ACTIVE_MODEL_NAME = "osnet_x1_0"
_WARNED_ONCE = False


def _warn_once(message: str) -> None:
    global _WARNED_ONCE
    if _WARNED_ONCE:
        return
    _WARNED_ONCE = True
    logger.warning(message)


def _normalize_embedding(raw_embedding: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    if raw_embedding is None:
        return None
    arr = np.asarray(raw_embedding, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.isfinite(arr).all():
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return None
    return (arr / norm).astype(np.float32)


def _model_name_for_tier(model_tier: str) -> str:
    tier = str(model_tier or "").strip().lower()
    if tier == "fast":
        return "osnet_x0_5"
    if tier in {"high_accuracy", "high-accuracy", "high accuracy"}:
        return "osnet_x1_0"
    return "osnet_x1_0"


def _resolve_device(device_pref: str) -> str:
    pref = str(device_pref or "").strip().lower()
    if pref in {"cuda", "gpu"} and torch.cuda.is_available():
        return "cuda"
    if pref in {"cpu"}:
        return "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def initialize_reid(model_tier: str = "balanced", device_pref: str = "auto") -> str:
    """Initialize torchreid once and return backend label."""
    global _EXTRACTOR, _INIT_FAILED, _BACKEND_LABEL, _ACTIVE_MODEL_TIER, _ACTIVE_MODEL_NAME

    model_name = _model_name_for_tier(model_tier)
    tier = str(model_tier or "balanced").strip().lower()
    if tier not in {"fast", "balanced", "high_accuracy"}:
        tier = "balanced"

    with _LOCK:
        if _EXTRACTOR is not None and _ACTIVE_MODEL_TIER == tier:
            return _BACKEND_LABEL
        if _INIT_FAILED and _EXTRACTOR is None and _ACTIVE_MODEL_TIER == tier:
            return _BACKEND_LABEL

        if FeatureExtractor is None:
            _INIT_FAILED = True
            _BACKEND_LABEL = "disabled (torchreid unavailable)"
            _warn_once(f"torchreid import failed; Re-ID disabled: {_IMPORT_ERROR}")
            return _BACKEND_LABEL

        device = _resolve_device(device_pref)
        try:
            extractor = FeatureExtractor(
                model_name=model_name,
                model_path="",
                device=device,
            )
            _EXTRACTOR = extractor
            _INIT_FAILED = False
            _ACTIVE_MODEL_TIER = tier
            _ACTIVE_MODEL_NAME = model_name
            _BACKEND_LABEL = f"{device}:{model_name}"
            return _BACKEND_LABEL
        except Exception as exc:
            _EXTRACTOR = None
            _INIT_FAILED = True
            _ACTIVE_MODEL_TIER = tier
            _ACTIVE_MODEL_NAME = model_name
            _BACKEND_LABEL = f"disabled (init failed: {exc})"
            _warn_once(f"torchreid initialization failed; Re-ID disabled: {exc}")
            return _BACKEND_LABEL


def active_backend_label() -> str:
    return _BACKEND_LABEL


def extract_reid_embedding(face_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Extract one normalized Re-ID embedding from a BGR face crop."""
    if face_bgr is None or not isinstance(face_bgr, np.ndarray) or face_bgr.size == 0:
        return None

    if _EXTRACTOR is None:
        initialize_reid(_ACTIVE_MODEL_TIER, "auto")
    if _EXTRACTOR is None:
        return None

    try:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None

    try:
        features = _EXTRACTOR([rgb])  # type: ignore[misc,operator]
    except Exception as exc:
        _warn_once(f"torchreid feature extraction failed; disabling Re-ID extraction: {exc}")
        return None

    if hasattr(features, "detach"):
        try:
            features = features.detach().cpu().numpy()
        except Exception:
            return None
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] >= 1:
        arr = arr[0]
    return _normalize_embedding(arr)


def reid_similarity(emb1: Sequence[float], emb2: Sequence[float]) -> float:
    a = _normalize_embedding(emb1)
    b = _normalize_embedding(emb2)
    if a is None or b is None:
        return -1.0
    denom = (float(np.linalg.norm(a)) * float(np.linalg.norm(b))) + 1e-8
    return float(np.dot(a, b) / denom)
