#!/usr/bin/env python3
"""
Sort videos by female identity using facenet-pytorch on AMD ROCm.

Requirements:
- PyTorch built with ROCm support
- facenet-pytorch
- OpenCV
- scikit-learn
- tqdm
- Pillow

This script:
1. Recursively scans the source folder, including child folders.
2. Scans the first 60 seconds of each video using OpenCV.
3. Samples one frame every 2 seconds until it finds the first female face.
4. Uses InsightFace (RetinaFace + ArcFace) when enabled.
5. Falls back to legacy MTCNN + InceptionResnetV1 (FaceNet) if needed.
6. Uses a lightweight OpenCV DNN gender classifier on CPU.
7. Tracks the first detected female for 5-10 seconds and keeps only
   embeddings that match the same identity by cosine similarity.
8. Clusters video identities with DBSCAN.
9. Moves videos into Female_1, Female_2, ... or No_Female_Found.

Example:
    python sort_videos_by_female_faces_gpu.py \
      --input-dir /path/to/videos \
      --output-dir /path/to/sorted
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import DBSCAN
from tqdm import tqdm

try:
    # Legacy FaceNet path is preserved for backward-compatible fallback.
    from facenet_pytorch import InceptionResnetV1, MTCNN

    FACENET_AVAILABLE = True
    FACENET_IMPORT_ERROR: Optional[Exception] = None
except Exception as facenet_import_exc:
    InceptionResnetV1 = Any  # type: ignore[assignment,misc]
    MTCNN = Any  # type: ignore[assignment,misc]
    FACENET_AVAILABLE = False
    FACENET_IMPORT_ERROR = facenet_import_exc

try:
    from face_engine_insight import (
        compare_embeddings as insight_compare_embeddings,
        get_face_embedding as insight_get_face_embedding,
        get_faces as insight_get_faces,
        initialize_insightface as insight_initialize,
        provider_label as insight_provider_label,
    )

    INSIGHTFACE_MODULE_AVAILABLE = True
    INSIGHTFACE_IMPORT_ERROR: Optional[Exception] = None
except Exception as insight_import_exc:
    INSIGHTFACE_MODULE_AVAILABLE = False
    INSIGHTFACE_IMPORT_ERROR = insight_import_exc

from duplicate_tools import apply_duplicate_move, scan_duplicates
from folder_naming import build_cluster_folder_names, sanitize_folder_name
from identity_tools import list_identities, perform_identity_action
from learning_memory import (
    build_learning_summary,
    default_memory_path,
    load_memory,
    match_identity,
    record_feedback,
    save_memory,
)
from review_queue import (
    REVIEW_PENDING_DIRNAME,
    REVIEW_STATE_DIRNAME,
    apply_review_action,
    create_review_state,
    decide_review_route,
    load_review_state,
    pending_review_items,
    review_state_path,
    save_review_state,
)


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
GENDER_LABELS = ["Male", "Female"]
GENDER_PROTO_URL = (
    "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt"
)
GENDER_MODEL_URLS = [
    "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_net.caffemodel",
    "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_net.caffemodel",
]
GENDER_MEAN = (78.4263377603, 87.7689143744, 114.895847746)

LOGGER = logging.getLogger(__name__)

# Feature flag requested by integration plan; keeps old FaceNet path available.
USE_INSIGHTFACE = True


@dataclass
class Config:
    input_dir: str
    output_dir: str
    profile: str = "balanced"
    review_mode: bool = False
    reprocess_uncertain_only: bool = False
    learning_enabled: bool = True
    review_confidence_threshold: float = 0.75
    learning_memory_file: str = ""
    learning_auto_threshold: float = 0.82
    learning_suggest_threshold: float = 0.74
    stop_flag_file: str = ""
    recursive: bool = True
    include_generated_folders: bool = False
    live_trace: bool = False
    force_gpu: bool = False
    use_insightface: bool = USE_INSIGHTFACE
    max_seconds: int = 60
    sample_every_sec: float = 2.0
    resize_width: int = 960
    detection_batch_size: int = 4
    stabilization_seconds: float = 8.0
    stabilization_sample_sec: float = 1.0
    female_confirmation_frames: int = 2
    female_confirmation_window_sec: float = 2.5
    min_female_gender_confidence: float = 0.65
    min_female_vote_ratio: float = 0.62
    min_face_area_ratio: float = 0.018
    min_stable_embeddings: int = 3
    min_stabilization_gender_votes: int = 3
    mtcnn_min_face_size: int = 48
    mtcnn_margin_px: int = 16
    max_faces_per_frame: int = 4
    detection_confidence: float = 0.90
    same_person_threshold: float = 0.72
    duplicate_threshold: float = 0.985
    dbscan_eps: float = 0.35
    dbscan_min_samples: int = 1
    cluster_merge_threshold: float = 0.78
    max_workers: int = max(1, min(2, (os.cpu_count() or 4) // 2))
    gender_model_dir: str = ""
    gender_proto_path: str = ""
    gender_model_path: str = ""


CFG: Optional[Config] = None
DEVICE: Optional[torch.device] = None
MTCNN_MODEL: Optional[MTCNN] = None
EMBED_MODEL: Optional[InceptionResnetV1] = None
GENDER_NET: Optional[cv2.dnn_Net] = None
ACTIVE_ACCELERATION = "CPU fallback"
INSIGHTFACE_ACTIVE = False
INSIGHTFACE_BACKEND_LABEL = ""
PREVIEW_ENABLED = False
PREVIEW_WARNED = False
PREVIEW_WINDOW_TITLE = "Live Frame Preview"
RESULT_JSON_PREFIX = "[RESULT_JSON] "
REPORTS_DIRNAME = ".reports"
UNCERTAIN_DIRNAME = "Uncertain"
LEARNING_DIRNAME = ".learning"
SORTING_EMBEDDING_CACHE_FILENAME = "video_embedding_cache.json"
SORTING_EMBEDDING_CACHE_INSIGHTFACE_FILENAME = "video_embedding_cache_insightface.json"
SORTING_EMBEDDING_CACHE_FACENET_FILENAME = "video_embedding_cache_facenet.json"
SORTING_EMBEDDING_CACHE_SCHEMA_VERSION = 1
SORTING_EMBEDDING_CACHE_MAX_ITEMS = 50000
LEARNING_MEMORY_INSIGHTFACE_FILENAME = "memory_insightface_v1.json"
LEARNING_MEMORY_FACENET_FILENAME = "memory_facenet_v1.json"
RETRY_CHECKPOINT_RATIOS: Tuple[float, ...] = tuple(step / 100.0 for step in range(5, 100, 5))
REASON_TAG_PRIORITY = [
    "memory_match_applied",
    "gender_disagreement",
    "few_stable_embeddings",
    "few_votes",
    "low_face_area",
    "unverified_female_candidates",
    "memory_match_suggested",
]
PROFILE_FAST = "fast"
PROFILE_BALANCED = "balanced"
PROFILE_HIGH_ACCURACY = "high_accuracy"
PROFILE_KEY_TO_FLAG = {
    "max_seconds": "--max-seconds",
    "sample_every_sec": "--sample-every-sec",
    "stabilization_seconds": "--stabilization-seconds",
    "resize_width": "--resize-width",
    "detection_batch_size": "--detection-batch-size",
    "female_confirmation_frames": "--female-confirmation-frames",
    "min_female_vote_ratio": "--min-female-vote-ratio",
    "min_stable_embeddings": "--min-stable-embeddings",
    "min_stabilization_gender_votes": "--min-stabilization-gender-votes",
    "same_person_threshold": "--same-person-threshold",
    "cluster_merge_threshold": "--cluster-merge-threshold",
}
PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    PROFILE_FAST: {
        "max_seconds": 40,
        "sample_every_sec": 2.5,
        "stabilization_seconds": 6.0,
        "resize_width": 720,
        "detection_batch_size": 6,
        "female_confirmation_frames": 1,
        "min_female_vote_ratio": 0.58,
        "min_stable_embeddings": 2,
        "min_stabilization_gender_votes": 2,
        "same_person_threshold": 0.70,
        "cluster_merge_threshold": 0.76,
    },
    PROFILE_BALANCED: {
        "max_seconds": 60,
        "sample_every_sec": 2.0,
        "stabilization_seconds": 8.0,
        "resize_width": 960,
        "detection_batch_size": 4,
        "female_confirmation_frames": 2,
        "min_female_vote_ratio": 0.62,
        "min_stable_embeddings": 3,
        "min_stabilization_gender_votes": 3,
        "same_person_threshold": 0.72,
        "cluster_merge_threshold": 0.78,
    },
    PROFILE_HIGH_ACCURACY: {
        "max_seconds": 90,
        "sample_every_sec": 1.0,
        "stabilization_seconds": 10.0,
        "resize_width": 1152,
        "detection_batch_size": 3,
        "female_confirmation_frames": 3,
        "min_female_vote_ratio": 0.66,
        "min_stable_embeddings": 4,
        "min_stabilization_gender_votes": 4,
        "same_person_threshold": 0.74,
        "cluster_merge_threshold": 0.82,
    },
}

INSIGHTFACE_TUNED_DEFAULTS: Dict[str, Tuple[str, float]] = {
    # InsightFace/ArcFace score distribution differs from FaceNet defaults.
    "detection_confidence": ("--detection-confidence", 0.60),
    "same_person_threshold": ("--same-person-threshold", 0.45),
    "duplicate_threshold": ("--duplicate-threshold", 0.95),
    "dbscan_eps": ("--dbscan-eps", 0.40),
    "cluster_merge_threshold": ("--cluster-merge-threshold", 0.68),
}

FindFirstFemaleResult = Tuple[
    Optional[float],
    Optional[Tuple[int, int, int, int]],
    Optional[np.ndarray],
    Optional[str],
    Optional[float],
    Dict[str, Any],
]


class SortedEmbeddingCacheEntry(TypedDict):
    video_path: str
    source_video_path: str
    predicted_label: str
    decision_label: str
    confidence_score: float
    embedding: List[float]
    updated_at: str


class StopRequestedError(Exception):
    pass


def _opencv_gui_available() -> bool:
    try:
        info = cv2.getBuildInformation()
    except Exception:
        return False
    for line in info.splitlines():
        if "GUI:" in line:
            return "NONE" not in line.upper()
    return False


def embedding_model_key(use_insightface: bool) -> str:
    return "insightface" if bool(use_insightface) else "facenet"


def sorting_embedding_cache_filename(use_insightface: bool) -> str:
    return (
        SORTING_EMBEDDING_CACHE_INSIGHTFACE_FILENAME
        if bool(use_insightface)
        else SORTING_EMBEDDING_CACHE_FACENET_FILENAME
    )


def normalize_profile_name(raw_value: str) -> str:
    normalized = str(raw_value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {PROFILE_FAST, PROFILE_BALANCED, PROFILE_HIGH_ACCURACY}:
        return normalized
    return PROFILE_BALANCED


def apply_profile_defaults(args: argparse.Namespace, cli_tokens: Optional[Sequence[str]] = None) -> str:
    profile = normalize_profile_name(getattr(args, "profile", PROFILE_BALANCED))
    token_set = set(cli_tokens or [])
    for key, value in PROFILE_PRESETS.get(profile, PROFILE_PRESETS[PROFILE_BALANCED]).items():
        flag = PROFILE_KEY_TO_FLAG.get(key, "")
        if flag and flag in token_set:
            continue
        setattr(args, key, value)
    args.profile = profile
    return profile


def apply_uncertain_reprocess_defaults(args: argparse.Namespace, cli_tokens: Optional[Sequence[str]] = None) -> str:
    token_set = set(cli_tokens or [])
    if "--profile" not in token_set:
        args.profile = PROFILE_HIGH_ACCURACY
        for key, value in PROFILE_PRESETS[PROFILE_HIGH_ACCURACY].items():
            flag = PROFILE_KEY_TO_FLAG.get(key, "")
            if flag and flag in token_set:
                continue
            setattr(args, key, value)
    else:
        args.profile = normalize_profile_name(getattr(args, "profile", PROFILE_BALANCED))

    strict_defaults: Dict[str, Any] = {
        "female_confirmation_frames": 4,
        "min_stable_embeddings": 5,
        "min_stabilization_gender_votes": 5,
        "min_female_vote_ratio": 0.70,
    }
    for key, value in strict_defaults.items():
        flag = PROFILE_KEY_TO_FLAG.get(key, "")
        if flag and flag in token_set:
            continue
        setattr(args, key, value)
    return normalize_profile_name(getattr(args, "profile", PROFILE_BALANCED))


def apply_insightface_defaults(args: argparse.Namespace, cli_tokens: Optional[Sequence[str]] = None) -> None:
    """Apply ArcFace-friendly defaults while respecting explicit CLI overrides."""
    if not bool(getattr(args, "use_insightface", USE_INSIGHTFACE)):
        return

    token_set = set(cli_tokens or [])
    for key, (flag, value) in INSIGHTFACE_TUNED_DEFAULTS.items():
        if flag in token_set:
            continue
        setattr(args, key, value)


def gpu_smoke_test() -> Tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "torch.cuda is unavailable, so CPU will be used."

    try:
        device_name = torch.cuda.get_device_name(0)
        x = torch.randn((8, 8), device="cuda")
        y = x @ x
        _ = y.sum().item()
        return True, f"GPU smoke test passed on {device_name}"
    except Exception:
        return False, f"GPU smoke test failed: {traceback.format_exc(limit=1).strip()}"


def choose_preferred_device(force_gpu: bool = False) -> Tuple[torch.device, str]:
    if not torch.cuda.is_available():
        return torch.device("cpu"), "torch.cuda is unavailable, so CPU will be used."

    if force_gpu:
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = "Unknown GPU"
        return torch.device("cuda"), f"GPU use forced by flag on {device_name}"

    smoke_ok, reason = gpu_smoke_test()
    if smoke_ok:
        return torch.device("cuda"), reason

    return torch.device("cpu"), reason


def get_runtime_info() -> Dict[str, Any]:
    chosen_device, reason = choose_preferred_device(force_gpu=False)
    info: Dict[str, Any] = {
        "torch_version": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": chosen_device.type,
        "device_count": 0,
        "device_names": [],
        "acceleration": "CPU fallback" if chosen_device.type == "cpu" else "GPU via ROCm",
        "reason": reason,
        "rocm_smi_available": False,
    }

    if torch.cuda.is_available():
        info["device_count"] = int(torch.cuda.device_count())
        info["device_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if chosen_device.type == "cuda" and info["hip_version"]:
            info["acceleration"] = "GPU via ROCm"
        elif chosen_device.type == "cuda":
            info["acceleration"] = "GPU via CUDA-compatible torch backend"

    for cmd in (["rocm-smi"], ["/opt/rocm/bin/rocm-smi"]):
        try:
            subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=5)
            info["rocm_smi_available"] = True
            break
        except Exception:
            continue

    return info


def print_gpu_status() -> None:
    info = get_runtime_info()

    print("\n===== GPU Status =====")
    print(f"PyTorch version: {info['torch_version']}")
    print(f"ROCm / HIP version: {info['hip_version']}")
    print(f"CUDA API available: {info['cuda_available']}")
    print(f"Selected acceleration: {info['acceleration']}")

    if info["cuda_available"]:
        print(f"Visible GPU devices: {info['device_count']}")
        for idx, name in enumerate(info["device_names"]):
            print(f"  [{idx}] {name}")
        print("Using device string: cuda")
    else:
        print(f"Reason: {info['reason']}")

    for cmd in (["rocm-smi"], ["/opt/rocm/bin/rocm-smi"]):
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=5)
            print("rocm-smi:")
            for line in out.strip().splitlines()[:20]:
                print(f"  {line}")
            break
        except Exception:
            continue
    else:
        print("rocm-smi not available in PATH.")

    print("======================\n")


def should_fallback_to_cpu(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "invalid device function",
            "hip error",
            "no kernel image is available",
            "device-side assert",
            "device not available",
        )
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists() and dst.stat().st_size > 0:
        return
    print(f"Downloading model asset: {dst.name}")
    tmp_dst = dst.with_suffix(dst.suffix + ".part")
    if tmp_dst.exists():
        tmp_dst.unlink()
    try:
        urllib.request.urlretrieve(url, tmp_dst)
        tmp_dst.replace(dst)
    except Exception:
        if tmp_dst.exists():
            tmp_dst.unlink()
        raise


def download_first_available(urls: Sequence[str], dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        return

    last_error: Optional[Exception] = None
    for url in urls:
        try:
            download_file(url, dst)
            return
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Failed to download {dst.name} from all known sources. Last error: {last_error}")


def resolve_gender_model_paths(cfg: Config) -> Tuple[Path, Path]:
    base_dir = Path(cfg.gender_model_dir or (Path(cfg.output_dir) / ".model_cache"))
    proto_path = Path(cfg.gender_proto_path) if cfg.gender_proto_path else base_dir / "gender_deploy.prototxt"
    model_path = Path(cfg.gender_model_path) if cfg.gender_model_path else base_dir / "gender_net.caffemodel"

    if not proto_path.exists():
        download_file(GENDER_PROTO_URL, proto_path)
    if not model_path.exists():
        download_first_available(GENDER_MODEL_URLS, model_path)

    return proto_path, model_path


def _load_legacy_facenet_models(cfg: Config, preferred_device: torch.device) -> str:
    """Legacy FaceNet loader kept for backward-compatible fallback."""
    global DEVICE, MTCNN_MODEL, EMBED_MODEL

    if not FACENET_AVAILABLE:
        raise RuntimeError(f"facenet-pytorch is unavailable: {FACENET_IMPORT_ERROR}")

    def _load_models(device: torch.device) -> str:
        global DEVICE, MTCNN_MODEL, EMBED_MODEL
        DEVICE = device
        MTCNN_MODEL = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=cfg.mtcnn_min_face_size,
            thresholds=[0.6, 0.7, 0.8],
            factor=0.709,
            post_process=False,
            keep_all=True,
            device=DEVICE,
        )
        EMBED_MODEL = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
        with torch.inference_mode():
            dummy = torch.zeros((1, 3, 160, 160), device=DEVICE)
            EMBED_MODEL(dummy)
        return "GPU via ROCm" if device.type == "cuda" else "CPU fallback"

    try:
        return _load_models(preferred_device)
    except Exception as exc:
        if preferred_device.type == "cuda" and should_fallback_to_cpu(exc):
            print(f"[WARN] FaceNet GPU initialization failed, falling back to CPU: {exc}", flush=True)
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            return _load_models(torch.device("cpu"))
        raise


def _is_insightface_enabled() -> bool:
    return bool(CFG and CFG.use_insightface and INSIGHTFACE_ACTIVE and INSIGHTFACE_MODULE_AVAILABLE)


def _disable_insightface_runtime(reason: Exception) -> None:
    """Disable InsightFace on runtime errors and attempt legacy FaceNet fallback."""
    global INSIGHTFACE_ACTIVE, INSIGHTFACE_BACKEND_LABEL, ACTIVE_ACCELERATION

    if not INSIGHTFACE_ACTIVE:
        return

    print(f"[WARN] InsightFace runtime failed, switching to FaceNet fallback: {reason}", flush=True)
    INSIGHTFACE_ACTIVE = False
    INSIGHTFACE_BACKEND_LABEL = ""

    if CFG is None:
        raise RuntimeError("Cannot fallback to FaceNet: runtime config is unavailable.") from reason

    preferred_device, preferred_reason = choose_preferred_device(force_gpu=CFG.force_gpu)
    print(f"[INFO] FaceNet fallback preferred acceleration: {preferred_device.type} | {preferred_reason}", flush=True)
    ACTIVE_ACCELERATION = _load_legacy_facenet_models(CFG, preferred_device) + " | FaceNet fallback"


def init_worker(cfg: Config) -> None:
    global CFG, DEVICE, MTCNN_MODEL, EMBED_MODEL, GENDER_NET, ACTIVE_ACCELERATION
    global PREVIEW_ENABLED, INSIGHTFACE_ACTIVE, INSIGHTFACE_BACKEND_LABEL
    CFG = cfg
    PREVIEW_ENABLED = bool(cfg.live_trace and cfg.max_workers == 1)
    INSIGHTFACE_ACTIVE = False
    INSIGHTFACE_BACKEND_LABEL = ""
    DEVICE = None
    MTCNN_MODEL = None
    EMBED_MODEL = None

    if PREVIEW_ENABLED and not _opencv_gui_available():
        PREVIEW_ENABLED = False
        print(
            "[WARN] Live preview disabled: OpenCV has no GUI support (likely opencv-python-headless). "
            "Fix: python -m pip uninstall -y opencv-python-headless && "
            "python -m pip install --upgrade opencv-python",
            flush=True,
        )

    proto_path, model_path = resolve_gender_model_paths(cfg)

    GENDER_NET = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
    GENDER_NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    GENDER_NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    preferred_device, preferred_reason = choose_preferred_device(force_gpu=cfg.force_gpu)
    print(f"[INFO] Preferred acceleration: {preferred_device.type} | {preferred_reason}", flush=True)

    if cfg.use_insightface:
        if not INSIGHTFACE_MODULE_AVAILABLE:
            print(f"[WARN] InsightFace import failed, using FaceNet fallback: {INSIGHTFACE_IMPORT_ERROR}", flush=True)
        else:
            try:
                INSIGHTFACE_BACKEND_LABEL = insight_initialize()
                INSIGHTFACE_ACTIVE = True
                ACTIVE_ACCELERATION = INSIGHTFACE_BACKEND_LABEL
                print(f"[INFO] InsightFace initialized: {INSIGHTFACE_BACKEND_LABEL}", flush=True)
                print(f"[INFO] InsightFace providers: {insight_provider_label()}", flush=True)
                return
            except Exception as exc:
                print(f"[WARN] InsightFace initialization failed, using FaceNet fallback: {exc}", flush=True)
                LOGGER.exception("InsightFace init failure", exc_info=exc)

    ACTIVE_ACCELERATION = _load_legacy_facenet_models(cfg, preferred_device) + " | FaceNet"
    print(f"[INFO] FaceNet initialized: {ACTIVE_ACCELERATION}", flush=True)


def list_videos(input_dir: str, recursive: bool = True, include_generated_folders: bool = False) -> List[Path]:
    root = Path(input_dir)
    iterator = root.rglob("*") if recursive else root.glob("*")
    videos: List[Path] = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        rel_parts = path.relative_to(root).parts
        rel_parts_set = set(rel_parts)
        if REVIEW_STATE_DIRNAME in rel_parts_set or REVIEW_PENDING_DIRNAME in rel_parts_set:
            continue
        if "Duplicates" in rel_parts_set or UNCERTAIN_DIRNAME in rel_parts_set:
            continue
        if not include_generated_folders:
            if ".model_cache" in rel_parts_set or "No_Female_Found" in rel_parts_set:
                continue
            if any(part.startswith("Female_") for part in rel_parts):
                continue
        videos.append(path)
    return sorted(videos)


def resize_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= target_width:
        return frame
    scale = target_width / float(width)
    return cv2.resize(frame, (target_width, int(height * scale)), interpolation=cv2.INTER_AREA)


def timestamp_to_frame_index(timestamp_sec: float, fps: float, total_frames: int) -> int:
    return max(0, min(total_frames - 1, int(timestamp_sec * fps)))


def read_frame_at(cap: cv2.VideoCapture, frame_index: int) -> Optional[np.ndarray]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def read_frames_at_indices(cap: cv2.VideoCapture, frame_indices: Sequence[int]) -> Dict[int, Optional[np.ndarray]]:
    if not frame_indices:
        return {}

    ordered = sorted(dict.fromkeys(int(idx) for idx in frame_indices))
    frames: Dict[int, Optional[np.ndarray]] = {}

    current_target = ordered[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_target)
    ok, frame = cap.read()
    frames[current_target] = frame if ok and frame is not None else None
    current_pos = current_target

    for target in ordered[1:]:
        gap = max(0, target - current_pos - 1)
        for _ in range(gap):
            if not cap.grab():
                break
        ok, frame = cap.read()
        frames[target] = frame if ok and frame is not None else None
        current_pos = target

    return frames


def get_video_meta(cap: cv2.VideoCapture) -> Tuple[float, int, float]:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if total_frames > 0 else 0.0
    return fps, total_frames, duration_sec


def clamp_box(box: Sequence[float], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width - 1, int(round(x2))))
    y2 = max(0, min(height - 1, int(round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def expand_box(box: Tuple[int, int, int, int], margin_px: int, width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - margin_px),
        max(0, y1 - margin_px),
        min(width, x2 + margin_px),
        min(height, y2 + margin_px),
    )


def crop_bgr(frame_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = box
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def face_area_ratio(box: Tuple[int, int, int, int], frame_bgr: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    face_area = max(0, x2 - x1) * max(0, y2 - y1)
    frame_area = max(1, frame_bgr.shape[0] * frame_bgr.shape[1])
    return float(face_area) / float(frame_area)


def detect_faces(frame_rgb: np.ndarray) -> List[Dict[str, Any]]:
    if _is_insightface_enabled():
        try:
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            return insight_get_faces(frame_bgr)
        except Exception as exc:
            _disable_insightface_runtime(exc)
    return _detect_faces_single(frame_rgb)


def _format_detections_for_frame(
    boxes: Any,
    probs: Any,
    frame_rgb: np.ndarray,
) -> List[Dict[str, Any]]:
    if boxes is None or probs is None:
        return []

    frame_h, frame_w = frame_rgb.shape[:2]
    detections: List[Dict[str, Any]] = []
    for box, prob in zip(boxes, probs):
        if box is None or prob is None:
            continue
        clamped = clamp_box(box, frame_w, frame_h)
        if clamped is None:
            continue
        x1, y1, x2, y2 = clamped
        detections.append(
            {
                "bbox": clamped,
                "prob": float(prob),
                "area": float((x2 - x1) * (y2 - y1)),
                "embedding": None,
            }
        )

    detections.sort(key=lambda item: (item["area"] * item["prob"]), reverse=True)
    return detections


def _detect_faces_single(frame_rgb: np.ndarray) -> List[Dict[str, Any]]:
    assert MTCNN_MODEL is not None
    boxes, probs = MTCNN_MODEL.detect(Image.fromarray(frame_rgb))
    return _format_detections_for_frame(boxes, probs, frame_rgb)


def detect_faces_batch(frames_rgb: Sequence[np.ndarray]) -> List[List[Dict[str, Any]]]:
    if not frames_rgb:
        return []

    if _is_insightface_enabled():
        try:
            return [insight_get_faces(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)) for frame in frames_rgb]
        except Exception as exc:
            _disable_insightface_runtime(exc)

    assert MTCNN_MODEL is not None
    pil_frames = [Image.fromarray(frame) for frame in frames_rgb]
    try:
        boxes_batch, probs_batch = MTCNN_MODEL.detect(pil_frames if len(pil_frames) > 1 else pil_frames[0])
    except Exception:
        return [_detect_faces_single(frame) for frame in frames_rgb]

    if len(frames_rgb) == 1 and not isinstance(boxes_batch, (list, tuple)):
        boxes_seq = [boxes_batch]
        probs_seq = [probs_batch]
    else:
        boxes_seq = boxes_batch
        probs_seq = probs_batch

    return [
        _format_detections_for_frame(boxes, probs, frame_rgb)
        for frame_rgb, boxes, probs in zip(frames_rgb, boxes_seq, probs_seq)
    ]


def classify_gender(face_bgr: np.ndarray) -> Tuple[str, float]:
    assert GENDER_NET is not None

    blob = cv2.dnn.blobFromImage(
        image=cv2.resize(face_bgr, (227, 227)),
        scalefactor=1.0,
        size=(227, 227),
        mean=GENDER_MEAN,
        swapRB=False,
        crop=False,
    )
    GENDER_NET.setInput(blob)
    probs = GENDER_NET.forward()[0]
    best_idx = int(np.argmax(probs))
    return GENDER_LABELS[best_idx], float(probs[best_idx])


def prepare_face_tensor(face_rgb: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(face_rgb, (160, 160), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float()
    return (tensor - 127.5) / 128.0


def _normalize_embedding_vector(embedding: Sequence[float]) -> Optional[np.ndarray]:
    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return None
    denom = float(np.linalg.norm(arr))
    if denom <= 1e-8:
        return None
    return (arr / denom).astype(np.float32)


def _embed_faces_facenet(face_crops_rgb: Sequence[np.ndarray]) -> np.ndarray:
    # Legacy FaceNet embedding path retained for backward compatibility.
    assert EMBED_MODEL is not None
    assert DEVICE is not None

    tensors = [prepare_face_tensor(face) for face in face_crops_rgb if face is not None and face.size > 0]
    if not tensors:
        return np.empty((0, 512), dtype=np.float32)

    batch = torch.stack(tensors, dim=0).to(DEVICE, non_blocking=True)
    with torch.inference_mode():
        embeddings = EMBED_MODEL(batch)
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.detach().cpu().numpy().astype(np.float32)


def get_face_embedding(
    image_bgr: np.ndarray,
    precomputed_detection: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    """Return one normalized embedding and bbox from a BGR frame/crop."""
    if image_bgr is None or image_bgr.size == 0:
        return None, None

    if precomputed_detection is not None:
        existing = precomputed_detection.get("embedding")
        bbox = precomputed_detection.get("bbox")
        normalized_existing = _normalize_embedding_vector(existing) if existing is not None else None
        if normalized_existing is not None:
            return normalized_existing, bbox

    if _is_insightface_enabled():
        try:
            emb, bbox = insight_get_face_embedding(image_bgr)
            if emb is None:
                return None, bbox
            normalized = _normalize_embedding_vector(emb)
            return normalized, bbox
        except Exception as exc:
            _disable_insightface_runtime(exc)

    if EMBED_MODEL is None or DEVICE is None:
        return None, None

    # FaceNet fallback expects RGB tensor input.
    face_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    emb_batch = _embed_faces_facenet([face_rgb])
    if len(emb_batch) == 0:
        return None, None
    bbox = (0, 0, int(image_bgr.shape[1]), int(image_bgr.shape[0]))
    return emb_batch[0], bbox


def embed_faces(face_crops_rgb: Sequence[np.ndarray]) -> np.ndarray:
    if _is_insightface_enabled():
        embeddings: List[np.ndarray] = []
        for face_rgb in face_crops_rgb:
            if face_rgb is None or face_rgb.size == 0:
                continue
            face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
            emb, _ = get_face_embedding(face_bgr)
            if emb is not None:
                embeddings.append(emb)
        if not embeddings:
            return np.empty((0, 512), dtype=np.float32)
        return np.vstack(embeddings).astype(np.float32)

    return _embed_faces_facenet(face_crops_rgb)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if INSIGHTFACE_MODULE_AVAILABLE:
        try:
            return float(insight_compare_embeddings(a, b))
        except Exception:
            pass
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def build_sample_times(max_seconds: float, step_seconds: float) -> List[float]:
    times: List[float] = []
    t = 0.0
    while t <= max_seconds:
        times.append(t)
        t += step_seconds
    return times


def empty_scan_info() -> Dict[str, Any]:
    return {
        "female_seed_hits": 0,
        "best_seed_confidence": 0.0,
        "total_faces_evaluated": 0,
        "low_face_area_rejections": 0,
    }


def merge_scan_info(base: Optional[Dict[str, Any]], extra: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base_info = base or {}
    extra_info = extra or {}
    return {
        "female_seed_hits": int(base_info.get("female_seed_hits", 0)) + int(extra_info.get("female_seed_hits", 0)),
        "best_seed_confidence": max(
            float(base_info.get("best_seed_confidence", 0.0)),
            float(extra_info.get("best_seed_confidence", 0.0)),
        ),
        "total_faces_evaluated": int(base_info.get("total_faces_evaluated", 0))
        + int(extra_info.get("total_faces_evaluated", 0)),
        "low_face_area_rejections": int(base_info.get("low_face_area_rejections", 0))
        + int(extra_info.get("low_face_area_rejections", 0)),
    }


def has_verified_female_candidate(
    first_ts: Optional[float],
    first_box: Optional[Tuple[int, int, int, int]],
    first_embedding: Optional[np.ndarray],
    first_gender_label: Optional[str],
    first_gender_conf: Optional[float],
) -> bool:
    return (
        first_ts is not None
        and first_box is not None
        and first_embedding is not None
        and first_gender_label is not None
        and first_gender_conf is not None
    )


def build_uncertain_retry_checkpoints(
    *,
    seed_hits: int,
    duration_sec: float,
    initial_scan_end_sec: float,
    require_seed_hits: bool = True,
) -> List[float]:
    if require_seed_hits and seed_hits <= 0:
        return []

    total_duration = max(0.0, float(duration_sec))
    if total_duration <= 0:
        return []

    already_covered_until = max(0.0, min(float(initial_scan_end_sec), total_duration))
    if already_covered_until >= total_duration:
        return []

    checkpoints: List[float] = []
    seen: set[float] = set()
    for ratio in RETRY_CHECKPOINT_RATIOS:
        checkpoint = round(total_duration * float(ratio), 6)
        key = round(checkpoint, 6)
        if checkpoint <= already_covered_until + 1e-6:
            continue
        if checkpoint >= total_duration:
            continue
        if key in seen:
            continue
        seen.add(key)
        checkpoints.append(checkpoint)

    checkpoints.sort()
    return checkpoints


def retry_find_first_female(
    *,
    cap: cv2.VideoCapture,
    fps: float,
    total_frames: int,
    duration_sec: float,
    scan_window_sec: float,
    cfg: Config,
    video_name: str,
    checkpoint_starts: Sequence[float],
    scan_callable: Optional[Callable[..., FindFirstFemaleResult]] = None,
) -> Tuple[
    Optional[float],
    Optional[Tuple[int, int, int, int]],
    Optional[np.ndarray],
    Optional[str],
    Optional[float],
    Dict[str, Any],
    Optional[float],
    List[float],
]:
    scanner = scan_callable or find_first_female
    attempted_checkpoints: List[float] = []
    merged_info = empty_scan_info()

    for checkpoint_start in checkpoint_starts:
        retry_scan_end = min(float(duration_sec), float(checkpoint_start) + float(scan_window_sec))
        if retry_scan_end <= checkpoint_start:
            continue

        attempted_checkpoints.append(float(checkpoint_start))
        print(
            f"[RETRY] {video_name} retry scan at {checkpoint_start:.1f}s -> {retry_scan_end:.1f}s",
            flush=True,
        )
        first_ts, first_box, first_embedding, first_gender_label, first_gender_conf, retry_info = scanner(
            cap=cap,
            fps=fps,
            total_frames=total_frames,
            scan_duration=retry_scan_end,
            cfg=cfg,
            video_name=video_name,
            scan_start_sec=float(checkpoint_start),
        )
        merged_info = merge_scan_info(merged_info, retry_info)

        if has_verified_female_candidate(first_ts, first_box, first_embedding, first_gender_label, first_gender_conf):
            print(
                f"[RETRY] {video_name} recovered at checkpoint {checkpoint_start:.1f}s",
                flush=True,
            )
            return (
                first_ts,
                first_box,
                first_embedding,
                first_gender_label,
                first_gender_conf,
                merged_info,
                retry_scan_end,
                attempted_checkpoints,
            )

    return None, None, None, None, None, merged_info, None, attempted_checkpoints


def emit_trace(video_name: str, phase: str, timestamp_sec: float, frame_index: int) -> None:
    if CFG is None or not CFG.live_trace:
        return
    print(
        f"[TRACE] video={video_name} phase={phase} time={timestamp_sec:.2f}s frame={frame_index}",
        flush=True,
    )


def stop_requested() -> bool:
    if CFG is None:
        return False
    flag = str(CFG.stop_flag_file or "").strip()
    if not flag:
        return False
    return Path(flag).exists()


def ensure_not_stopped() -> None:
    if stop_requested():
        raise StopRequestedError("Stop requested")


def show_live_preview(
    frame_bgr: Optional[np.ndarray],
    video_name: str,
    phase: str,
    timestamp_sec: float,
    frame_index: int,
) -> None:
    global PREVIEW_ENABLED, PREVIEW_WARNED

    if CFG is None or not CFG.live_trace or not PREVIEW_ENABLED or frame_bgr is None:
        return

    try:
        preview = frame_bgr.copy()
        overlay = f"{phase} | {video_name} | {timestamp_sec:.2f}s | frame {frame_index}"
        cv2.putText(
            preview,
            overlay,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(PREVIEW_WINDOW_TITLE, preview)
        cv2.waitKey(1)
    except Exception as exc:
        PREVIEW_ENABLED = False
        if not PREVIEW_WARNED:
            PREVIEW_WARNED = True
            print(f"[WARN] Live preview unavailable, using text trace only: {exc}", flush=True)


def close_live_preview() -> None:
    if not PREVIEW_ENABLED:
        return
    try:
        cv2.destroyWindow(PREVIEW_WINDOW_TITLE)
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def emit_progress(done: int, total: int, female: int, no_female: int, errors: int) -> None:
    print(
        f"[PROGRESS] done={done} total={total} female={female} no_female={no_female} errors={errors}",
        flush=True,
    )


def safe_console_print(message: Any, *, flush: bool = False) -> None:
    text = str(message)
    try:
        print(text, flush=flush)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        line = text if text.endswith("\n") else f"{text}\n"
        payload = line.encode(encoding, errors="replace")
        buffer = getattr(sys.stdout, "buffer", None)
        if buffer is not None:
            buffer.write(payload)
            if flush:
                buffer.flush()
            return
        sys.stdout.write(payload.decode(encoding, errors="replace"))
        if flush:
            sys.stdout.flush()


def configure_console_encoding() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def robust_average_embeddings(embeddings: Sequence[np.ndarray]) -> np.ndarray:
    matrix = np.vstack(embeddings).astype(np.float32)
    center = np.mean(matrix, axis=0)
    center /= np.linalg.norm(center) + 1e-8
    sims = matrix @ center
    keep_count = max(1, int(np.ceil(len(matrix) * 0.8)))
    keep_indices = np.argsort(sims)[-keep_count:]
    stable = np.mean(matrix[keep_indices], axis=0).astype(np.float32)
    stable /= np.linalg.norm(stable) + 1e-8
    return stable


def verify_female_candidate(
    cap: cv2.VideoCapture,
    fps: float,
    total_frames: int,
    scan_duration: float,
    first_ts: float,
    first_embedding: np.ndarray,
    cfg: Config,
    initial_gender_label: str,
    initial_gender_conf: float,
) -> Tuple[bool, float, float, int]:
    ensure_not_stopped()
    female_score = initial_gender_conf if initial_gender_label.lower() == "female" else 0.0
    male_score = initial_gender_conf if initial_gender_label.lower() == "male" else 0.0
    female_votes = 1 if initial_gender_label.lower() == "female" else 0
    votes = 1

    step = max(0.5, cfg.female_confirmation_window_sec / max(1, cfg.female_confirmation_frames))
    for idx in range(1, cfg.female_confirmation_frames):
        ensure_not_stopped()
        ts = first_ts + (idx * step)
        if ts > scan_duration:
            break
        frame_index = timestamp_to_frame_index(ts, fps, total_frames)
        frame_bgr = read_frame_at(cap, frame_index)
        if frame_bgr is None:
            continue

        frame_bgr = resize_frame(frame_bgr, cfg.resize_width)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = detect_faces(frame_rgb)
        if not faces:
            continue

        matched_embedding, matched_box, similarity = select_best_matching_face(
            frame_bgr=frame_bgr,
            faces=faces,
            reference_embedding=first_embedding,
            cfg=cfg,
        )
        if matched_embedding is None or matched_box is None or similarity < cfg.same_person_threshold:
            continue

        face_bgr = crop_bgr(frame_bgr, matched_box)
        if face_bgr is None:
            continue
        if face_area_ratio(matched_box, frame_bgr) < cfg.min_face_area_ratio:
            continue

        gender_label, gender_conf = classify_gender(face_bgr)
        votes += 1
        if gender_label.lower() == "female":
            female_score += gender_conf
            if gender_conf >= cfg.min_female_gender_confidence:
                female_votes += 1
        else:
            male_score += gender_conf

    if votes < cfg.female_confirmation_frames:
        return False, female_score, male_score, votes

    min_female_votes = max(2, cfg.female_confirmation_frames - 1)
    if female_votes < min_female_votes:
        return False, female_score, male_score, votes

    total_score = female_score + male_score
    if total_score <= 0:
        return False, female_score, male_score, votes

    female_ratio = female_score / total_score
    if female_ratio < cfg.min_female_vote_ratio:
        return False, female_score, male_score, votes

    if female_score <= male_score:
        return False, female_score, male_score, votes

    return True, female_score, male_score, votes


def find_first_female(
    cap: cv2.VideoCapture,
    fps: float,
    total_frames: int,
    scan_duration: float,
    cfg: Config,
    video_name: str,
    scan_start_sec: float = 0.0,
) -> FindFirstFemaleResult:
    scan_start_sec = max(0.0, float(scan_start_sec))
    scan_end_sec = max(scan_start_sec, float(scan_duration))
    sample_times = [scan_start_sec + offset for offset in build_sample_times(scan_end_sec - scan_start_sec, cfg.sample_every_sec)]
    sample_indices = [timestamp_to_frame_index(ts, fps, total_frames) for ts in sample_times]
    female_seed_hits = 0
    best_seed_conf = 0.0
    total_faces_evaluated = 0
    low_face_area_rejections = 0

    for start in range(0, len(sample_times), cfg.detection_batch_size):
        ensure_not_stopped()
        chunk_times = sample_times[start : start + cfg.detection_batch_size]
        chunk_indices = sample_indices[start : start + cfg.detection_batch_size]
        frame_map = read_frames_at_indices(cap, chunk_indices)

        valid_items: List[Tuple[float, int, np.ndarray, np.ndarray]] = []
        for timestamp_sec, frame_index in zip(chunk_times, chunk_indices):
            emit_trace(video_name, "scan", timestamp_sec, frame_index)
            frame_bgr = frame_map.get(frame_index)
            if frame_bgr is None:
                continue
            frame_bgr = resize_frame(frame_bgr, cfg.resize_width)
            show_live_preview(frame_bgr, video_name, "scan", timestamp_sec, frame_index)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            valid_items.append((timestamp_sec, frame_index, frame_bgr, frame_rgb))

        if not valid_items:
            continue

        detections_batch = detect_faces_batch([item[3] for item in valid_items])
        for (timestamp_sec, _frame_index, frame_bgr, _frame_rgb), faces in zip(valid_items, detections_batch):
            if not faces:
                continue

            checked = 0
            for face in faces:
                if face["prob"] < cfg.detection_confidence:
                    continue
                if checked >= cfg.max_faces_per_frame:
                    break
                checked += 1
                total_faces_evaluated += 1

                expanded = expand_box(face["bbox"], cfg.mtcnn_margin_px, frame_bgr.shape[1], frame_bgr.shape[0])
                if face_area_ratio(expanded, frame_bgr) < cfg.min_face_area_ratio:
                    low_face_area_rejections += 1
                    continue
                face_bgr = crop_bgr(frame_bgr, expanded)
                if face_bgr is None:
                    continue

                gender_label, gender_conf = classify_gender(face_bgr)
                if gender_label.lower() != "female" or gender_conf < cfg.min_female_gender_confidence:
                    continue
                female_seed_hits += 1
                best_seed_conf = max(best_seed_conf, float(gender_conf))

                embedding, _ = get_face_embedding(face_bgr, precomputed_detection=face)
                if embedding is None:
                    continue

                verified, female_score, male_score, vote_count = verify_female_candidate(
                    cap=cap,
                    fps=fps,
                    total_frames=total_frames,
                    scan_duration=scan_end_sec,
                    first_ts=timestamp_sec,
                    first_embedding=embedding,
                    cfg=cfg,
                    initial_gender_label=gender_label,
                    initial_gender_conf=gender_conf,
                )
                if not verified:
                    continue

                print(
                    f"[SCAN] first female at {timestamp_sec:.1f}s "
                    f"(seed_conf={gender_conf:.3f}, female_score={female_score:.3f}, "
                    f"male_score={male_score:.3f}, votes={vote_count})",
                    flush=True,
                )
                return (
                    timestamp_sec,
                    expanded,
                    embedding,
                    gender_label,
                    gender_conf,
                    {
                        "female_seed_hits": female_seed_hits,
                        "best_seed_confidence": best_seed_conf,
                        "total_faces_evaluated": total_faces_evaluated,
                        "low_face_area_rejections": low_face_area_rejections,
                    },
                )

    return (
        None,
        None,
        None,
        None,
        None,
        {
            "female_seed_hits": female_seed_hits,
            "best_seed_confidence": best_seed_conf,
            "total_faces_evaluated": total_faces_evaluated,
            "low_face_area_rejections": low_face_area_rejections,
        },
    )


def select_best_matching_face(
    frame_bgr: np.ndarray,
    faces: List[Dict[str, Any]],
    reference_embedding: np.ndarray,
    cfg: Config,
    diagnostics: Optional[Dict[str, int]] = None,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], float]:
    candidate_boxes: List[Tuple[int, int, int, int]] = []
    candidate_embeddings: List[Optional[np.ndarray]] = []

    for face in faces[: cfg.max_faces_per_frame]:
        if face["prob"] < cfg.detection_confidence:
            continue
        if diagnostics is not None:
            diagnostics["total_faces_evaluated"] = int(diagnostics.get("total_faces_evaluated", 0)) + 1
        expanded = expand_box(face["bbox"], cfg.mtcnn_margin_px, frame_bgr.shape[1], frame_bgr.shape[0])
        if face_area_ratio(expanded, frame_bgr) < cfg.min_face_area_ratio:
            if diagnostics is not None:
                diagnostics["low_face_area_rejections"] = int(diagnostics.get("low_face_area_rejections", 0)) + 1
            continue
        crop = crop_bgr(frame_bgr, expanded)
        if crop is None:
            continue
        candidate_boxes.append(expanded)
        embedding, _ = get_face_embedding(crop, precomputed_detection=face)
        candidate_embeddings.append(embedding)

    if not candidate_embeddings:
        return None, None, -1.0

    scored: List[Tuple[np.ndarray, Tuple[int, int, int, int], float]] = []
    for emb, box in zip(candidate_embeddings, candidate_boxes):
        if emb is None:
            continue
        scored.append((emb, box, cosine_similarity(reference_embedding, emb)))

    if not scored:
        return None, None, -1.0

    best_idx = int(np.argmax([item[2] for item in scored]))
    best_embedding, best_box, best_similarity = scored[best_idx]
    return best_embedding, best_box, float(best_similarity)


def stabilize_identity(
    cap: cv2.VideoCapture,
    fps: float,
    total_frames: int,
    scan_duration: float,
    first_ts: float,
    first_box: Tuple[int, int, int, int],
    first_embedding: np.ndarray,
    initial_gender_label: str,
    initial_gender_conf: float,
    cfg: Config,
    video_name: str,
) -> Tuple[List[np.ndarray], float, float, int, Dict[str, int]]:
    ensure_not_stopped()
    diagnostics: Dict[str, int] = {
        "total_faces_evaluated": 0,
        "low_face_area_rejections": 0,
    }
    end_ts = min(scan_duration, first_ts + cfg.stabilization_seconds)
    if end_ts <= first_ts:
        female_score = initial_gender_conf if initial_gender_label.lower() == "female" else 0.0
        male_score = initial_gender_conf if initial_gender_label.lower() == "male" else 0.0
        return [first_embedding], female_score, male_score, 1, diagnostics

    kept_embeddings: List[np.ndarray] = [first_embedding]
    female_score = initial_gender_conf if initial_gender_label.lower() == "female" else 0.0
    male_score = initial_gender_conf if initial_gender_label.lower() == "male" else 0.0
    gender_votes = 1
    running_reference = first_embedding.copy()
    last_kept = first_embedding.copy()
    current_box = first_box

    sample_offsets = build_sample_times(end_ts - first_ts, cfg.stabilization_sample_sec)[1:]
    sample_times = [first_ts + offset_sec for offset_sec in sample_offsets]
    sample_indices = [timestamp_to_frame_index(ts, fps, total_frames) for ts in sample_times]

    for start in range(0, len(sample_times), cfg.detection_batch_size):
        ensure_not_stopped()
        chunk_times = sample_times[start : start + cfg.detection_batch_size]
        chunk_indices = sample_indices[start : start + cfg.detection_batch_size]
        frame_map = read_frames_at_indices(cap, chunk_indices)

        valid_items: List[Tuple[float, int, np.ndarray, np.ndarray]] = []
        for timestamp_sec, frame_index in zip(chunk_times, chunk_indices):
            emit_trace(video_name, "stabilize", timestamp_sec, frame_index)
            frame_bgr = frame_map.get(frame_index)
            if frame_bgr is None:
                continue
            frame_bgr = resize_frame(frame_bgr, cfg.resize_width)
            show_live_preview(frame_bgr, video_name, "stabilize", timestamp_sec, frame_index)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            valid_items.append((timestamp_sec, frame_index, frame_bgr, frame_rgb))

        if not valid_items:
            continue

        detections_batch = detect_faces_batch([item[3] for item in valid_items])
        for (_timestamp_sec, _frame_index, frame_bgr, _frame_rgb), faces in zip(valid_items, detections_batch):
            if not faces:
                continue

            matched_embedding, matched_box, similarity = select_best_matching_face(
                frame_bgr=frame_bgr,
                faces=faces,
                reference_embedding=running_reference,
                cfg=cfg,
                diagnostics=diagnostics,
            )
            if matched_embedding is None or matched_box is None:
                continue
            if similarity < cfg.same_person_threshold:
                continue

            box_center_shift = abs((matched_box[0] + matched_box[2]) - (current_box[0] + current_box[2])) + abs(
                (matched_box[1] + matched_box[3]) - (current_box[1] + current_box[3])
            )
            if box_center_shift > max(frame_bgr.shape[:2]) * 0.8:
                continue

            if cosine_similarity(last_kept, matched_embedding) >= cfg.duplicate_threshold:
                continue

            face_bgr = crop_bgr(frame_bgr, matched_box)
            if face_bgr is not None:
                gender_label, gender_conf = classify_gender(face_bgr)
                gender_votes += 1
                if gender_label.lower() == "female":
                    female_score += gender_conf
                else:
                    male_score += gender_conf

            kept_embeddings.append(matched_embedding)
            last_kept = matched_embedding
            current_box = matched_box
            running_reference = np.mean(np.vstack(kept_embeddings), axis=0).astype(np.float32)
            running_reference /= np.linalg.norm(running_reference) + 1e-8

    return kept_embeddings, female_score, male_score, gender_votes, diagnostics


def clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def set_decision(result: Dict[str, Any], label: str, reason: str, confidence: float) -> None:
    result["decision_label"] = label
    result["decision_reason"] = reason
    result["confidence_score"] = round(clamp_confidence(confidence), 3)


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


def new_result_record(video_path: str, device_label: str) -> Dict[str, Any]:
    return {
        "video": video_path,
        "female_found": False,
        "embedding": None,
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


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_reason_metrics(metrics: Any) -> Dict[str, Any]:
    if not isinstance(metrics, dict):
        metrics = {}
    normalized = empty_reason_metrics()
    normalized["total_faces_evaluated"] = max(0, _safe_int(metrics.get("total_faces_evaluated", 0)))
    normalized["low_face_area_rejections"] = max(0, _safe_int(metrics.get("low_face_area_rejections", 0)))
    normalized["stable_embeddings"] = max(0, _safe_int(metrics.get("stable_embeddings", 0)))
    normalized["gender_votes"] = max(0, _safe_int(metrics.get("gender_votes", 0)))
    normalized["female_seed_hits"] = max(0, _safe_int(metrics.get("female_seed_hits", 0)))
    normalized["female_score"] = round(max(0.0, _safe_float(metrics.get("female_score", 0.0))), 4)
    normalized["male_score"] = round(max(0.0, _safe_float(metrics.get("male_score", 0.0))), 4)
    normalized["best_seed_confidence"] = round(
        clamp_confidence(_safe_float(metrics.get("best_seed_confidence", 0.0))),
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


def apply_explainability_metadata(result: Dict[str, Any], cfg: Config) -> None:
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

    if (
        stable_embeddings > 0 and stable_embeddings < cfg.min_stable_embeddings
    ) or ("stable embeddings" in reason_lower) or ("identity could not be stabilized" in reason_lower):
        tags.append("few_stable_embeddings")

    if (gender_votes > 0 and gender_votes < cfg.min_stabilization_gender_votes) or (
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
    payload = {
        "video": video_path,
        "video_name": Path(video_path).name if video_path else "",
        "decision_label": str(result.get("decision_label", "")),
        "confidence_score": round(clamp_confidence(_safe_float(result.get("confidence_score", 0.0))), 3),
        "reason_summary": str(result.get("reason_summary", "")),
        "reason_tags": [str(tag) for tag in reason_tags],
        "reason_metrics": _normalize_reason_metrics(result.get("reason_metrics")),
        "decision_reason": str(result.get("decision_reason", "")),
        "memory_match_label": str(result.get("memory_match_label", "")),
        "memory_match_score": round(_safe_float(result.get("memory_match_score", 0.0)), 4),
        "memory_applied": bool(result.get("memory_applied", False)),
        "learning_applied": bool(result.get("learning_applied", result.get("memory_applied", False))),
        "adaptive_threshold_used": round(_safe_float(result.get("adaptive_threshold_used", 0.0)), 4),
        "feedback_consistency_snapshot": result.get("feedback_consistency_snapshot", {}),
        "stopped": bool(result.get("stopped", False)),
        "error": str(result.get("error") or ""),
    }
    return payload


def emit_result_json(result: Dict[str, Any]) -> None:
    payload = build_result_json_payload(result)
    print(f"{RESULT_JSON_PREFIX}{json.dumps(payload, separators=(',', ':'), ensure_ascii=True)}", flush=True)


def _atomic_write_text(path: Path, content: str) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def report_output_dir(output_dir: Path) -> Path:
    return output_dir / REPORTS_DIRNAME


def create_report_run_id(now_utc: datetime) -> str:
    return f"{now_utc.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def sorting_embedding_cache_path(output_dir: Path, use_insightface: bool) -> Path:
    return output_dir / LEARNING_DIRNAME / sorting_embedding_cache_filename(use_insightface)


def _normalize_cache_key(path_text: str) -> str:
    return os.path.normcase(os.path.normpath(str(path_text).strip()))


def _safe_embedding_vector(raw_embedding: Any) -> Optional[List[float]]:
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


def load_sorting_embedding_cache(path: Path) -> Dict[str, Any]:
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
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    for item in results:
        final_destination = str(item.get("final_destination", "")).strip()
        if not final_destination:
            continue
        embedding = _safe_embedding_vector(item.get("embedding"))
        if embedding is None:
            continue

        key = _normalize_cache_key(final_destination)
        payload: SortedEmbeddingCacheEntry = {
            "video_path": str(Path(final_destination).resolve()),
            "source_video_path": str(item.get("video", "")).strip(),
            "predicted_label": str(item.get("suggested_folder_name", "")).strip()
            or str(item.get("decision_label", "")).strip().lower(),
            "decision_label": str(item.get("decision_label", "")).strip().lower(),
            "confidence_score": round(clamp_confidence(_safe_float(item.get("confidence_score", 0.0))), 4),
            "embedding": embedding,
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
    _atomic_write_text(cache_path, json.dumps(cache, indent=2, ensure_ascii=False) + "\n")

    return {
        "path": str(cache_path),
        "updated_entries": int(updated),
        "total_entries": int(len(entries)),
    }


def build_per_video_report_rows(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in results:
        video_path = str(item.get("video", "")).strip()
        decision_label = str(item.get("decision_label", "")).strip().lower()
        rows.append(
            {
                "video_path": video_path,
                "video_name": Path(video_path).name if video_path else "",
                "decision_label": decision_label,
                "confidence_score": round(clamp_confidence(_safe_float(item.get("confidence_score", 0.0))), 3),
                "reason_summary": str(item.get("reason_summary", "")).strip(),
                "error": str(item.get("error") or "").strip(),
                "memory_applied": bool(item.get("memory_applied", False)),
                "final_destination": str(item.get("final_destination", "")).strip(),
            }
        )
    return rows


def build_run_summary_payload(
    *,
    run_id: str,
    run_started_at: datetime,
    run_finished_at: datetime,
    total_scanned: int,
    processed_successfully: int,
    female_detected: int,
    uncertain: int,
    no_female_found: int,
    errors: int,
    stopped_early: bool,
) -> Dict[str, Any]:
    time_taken_seconds = max(0.0, (run_finished_at - run_started_at).total_seconds())
    return {
        "run_id": run_id,
        "run_started_at": run_started_at.replace(microsecond=0).isoformat(),
        "run_finished_at": run_finished_at.replace(microsecond=0).isoformat(),
        "time_taken_seconds": round(time_taken_seconds, 3),
        "total_scanned": int(total_scanned),
        "processed_successfully": int(processed_successfully),
        "female_detected": int(female_detected),
        "uncertain": int(uncertain),
        "no_female_found": int(no_female_found),
        "errors": int(errors),
        "stopped_early": bool(stopped_early),
    }


def derive_decision_counts(results: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    female_detected = 0
    uncertain = 0
    no_female_found = 0
    errors = 0
    processed_successfully = 0
    for item in results:
        has_error = bool(item.get("error"))
        stopped = bool(item.get("stopped"))
        decision = str(item.get("decision_label", "")).strip().lower()
        if has_error or decision == "error":
            errors += 1
            continue
        if stopped or decision == "stopped":
            continue
        processed_successfully += 1
        if decision == "female_detected":
            female_detected += 1
        elif decision == "uncertain":
            uncertain += 1
        elif decision == "no_female":
            no_female_found += 1
    return {
        "processed_successfully": processed_successfully,
        "female_detected": female_detected,
        "uncertain": uncertain,
        "no_female_found": no_female_found,
        "errors": errors,
    }


def write_run_reports(
    *,
    output_dir: Path,
    run_id: str,
    summary_payload: Dict[str, Any],
    video_rows: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    reports_dir = report_output_dir(output_dir)
    ensure_dir(reports_dir)

    summary_json_path = reports_dir / f"run_{run_id}_summary.json"
    summary_csv_path = reports_dir / f"run_{run_id}_summary.csv"
    videos_json_path = reports_dir / f"run_{run_id}_videos.json"
    videos_csv_path = reports_dir / f"run_{run_id}_videos.csv"

    _atomic_write_text(summary_json_path, json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n")
    _atomic_write_text(videos_json_path, json.dumps(list(video_rows), indent=2, ensure_ascii=False) + "\n")

    summary_headers = [
        "run_id",
        "run_started_at",
        "run_finished_at",
        "time_taken_seconds",
        "total_scanned",
        "processed_successfully",
        "female_detected",
        "uncertain",
        "no_female_found",
        "errors",
        "stopped_early",
    ]
    videos_headers = [
        "video_path",
        "video_name",
        "decision_label",
        "confidence_score",
        "reason_summary",
        "error",
        "memory_applied",
        "final_destination",
    ]

    summary_io = io.StringIO()
    summary_writer = csv.DictWriter(summary_io, fieldnames=summary_headers, extrasaction="ignore")
    summary_writer.writeheader()
    summary_writer.writerow(summary_payload)
    _atomic_write_text(summary_csv_path, summary_io.getvalue())

    videos_io = io.StringIO()
    videos_writer = csv.DictWriter(videos_io, fieldnames=videos_headers, extrasaction="ignore")
    videos_writer.writeheader()
    for row in video_rows:
        videos_writer.writerow(row)
    _atomic_write_text(videos_csv_path, videos_io.getvalue())

    return {
        "summary_json": str(summary_json_path),
        "summary_csv": str(summary_csv_path),
        "videos_json": str(videos_json_path),
        "videos_csv": str(videos_csv_path),
    }


def emit_run_reports(
    *,
    output_dir: Path,
    run_id: str,
    run_started_at: datetime,
    run_finished_at: datetime,
    total_scanned: int,
    processed_successfully: int,
    female_detected: int,
    uncertain: int,
    no_female_found: int,
    errors: int,
    stopped_early: bool,
    results: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    summary_payload = build_run_summary_payload(
        run_id=run_id,
        run_started_at=run_started_at,
        run_finished_at=run_finished_at,
        total_scanned=total_scanned,
        processed_successfully=processed_successfully,
        female_detected=female_detected,
        uncertain=uncertain,
        no_female_found=no_female_found,
        errors=errors,
        stopped_early=stopped_early,
    )
    video_rows = build_per_video_report_rows(results)
    report_paths = write_run_reports(
        output_dir=output_dir,
        run_id=run_id,
        summary_payload=summary_payload,
        video_rows=video_rows,
    )
    print(
        "[REPORT] "
        f"summary_json={report_paths['summary_json']} "
        f"summary_csv={report_paths['summary_csv']} "
        f"videos_json={report_paths['videos_json']} "
        f"videos_csv={report_paths['videos_csv']}",
        flush=True,
    )
    return report_paths


def memory_file_path(cfg: Config) -> Path:
    explicit = str(cfg.learning_memory_file or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    base_default = default_memory_path(Path(__file__).resolve().parent)
    engine_filename = (
        LEARNING_MEMORY_INSIGHTFACE_FILENAME
        if bool(cfg.use_insightface)
        else LEARNING_MEMORY_FACENET_FILENAME
    )
    return base_default.with_name(engine_filename)


def apply_memory_assist(result: Dict[str, Any], memory: Dict[str, Any], cfg: Config) -> None:
    result["memory_match_label"] = ""
    result["memory_match_score"] = 0.0
    result["memory_applied"] = False
    result["learning_applied"] = False
    result["adaptive_threshold_used"] = round(float(cfg.learning_auto_threshold), 4)
    result["feedback_consistency_snapshot"] = {}
    result.setdefault("memory_suggestion", "")

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
        set_decision(
            result,
            "uncertain",
            (
                f"Memory suggests {label} (score={score:.3f}) but confidence is below "
                f"adaptive auto-apply threshold ({adaptive_auto_threshold:.3f})."
            ),
            max(base_conf, min(0.8, score)),
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


def process_video(video_path: str) -> Dict[str, Any]:
    assert CFG is not None
    cfg = CFG
    video_name = Path(video_path).name

    print(f"[START] {video_name}", flush=True)

    result: Dict[str, Any] = new_result_record(video_path, ACTIVE_ACCELERATION)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        result["error"] = "Failed to open video"
        set_decision(result, "error", "Failed to open video", 0.0)
        return result

    try:
        ensure_not_stopped()
        fps, total_frames, duration_sec = get_video_meta(cap)
        if total_frames <= 0:
            result["error"] = "Invalid or empty video"
            set_decision(result, "error", "Invalid or empty video", 0.0)
            return result

        scan_duration = min(float(cfg.max_seconds), duration_sec if duration_sec > 0 else float(cfg.max_seconds))
        active_scan_end = scan_duration
        first_ts, first_box, first_embedding, first_gender_label, first_gender_conf, scan_info = find_first_female(
            cap,
            fps,
            total_frames,
            active_scan_end,
            cfg,
            video_name,
        )
        scan_info = merge_scan_info(empty_scan_info(), scan_info)
        seed_hits = int(scan_info.get("female_seed_hits", 0))
        retry_duration_scope = duration_sec if cfg.reprocess_uncertain_only else scan_duration
        remaining_retry_checkpoints = build_uncertain_retry_checkpoints(
            seed_hits=seed_hits,
            duration_sec=retry_duration_scope,
            initial_scan_end_sec=scan_duration,
            require_seed_hits=not cfg.reprocess_uncertain_only,
        )

        def sync_seed_metrics() -> None:
            reason_metrics_inner = _normalize_reason_metrics(result.get("reason_metrics"))
            reason_metrics_inner["female_seed_hits"] = int(scan_info.get("female_seed_hits", 0))
            reason_metrics_inner["best_seed_confidence"] = round(float(scan_info.get("best_seed_confidence", 0.0)), 4)
            reason_metrics_inner["total_faces_evaluated"] = int(scan_info.get("total_faces_evaluated", 0))
            reason_metrics_inner["low_face_area_rejections"] = int(scan_info.get("low_face_area_rejections", 0))
            result["reason_metrics"] = reason_metrics_inner

        def recover_seed_from_retry(reason: str) -> bool:
            nonlocal first_ts, first_box, first_embedding, first_gender_label, first_gender_conf
            nonlocal active_scan_end, scan_info, remaining_retry_checkpoints
            if not remaining_retry_checkpoints:
                return False
            print(
                f"[RETRY] {video_name} {reason}; remaining checkpoints={len(remaining_retry_checkpoints)}",
                flush=True,
            )
            (
                retry_ts,
                retry_box,
                retry_embedding,
                retry_gender_label,
                retry_gender_conf,
                retry_scan_info,
                retry_scan_end,
                attempted_checkpoints,
            ) = retry_find_first_female(
                cap=cap,
                fps=fps,
                total_frames=total_frames,
                duration_sec=retry_duration_scope,
                scan_window_sec=scan_duration,
                cfg=cfg,
                video_name=video_name,
                checkpoint_starts=remaining_retry_checkpoints,
            )
            scan_info = merge_scan_info(scan_info, retry_scan_info)
            if attempted_checkpoints:
                consumed = min(len(remaining_retry_checkpoints), len(attempted_checkpoints))
                remaining_retry_checkpoints = remaining_retry_checkpoints[consumed:]
            sync_seed_metrics()
            if has_verified_female_candidate(
                retry_ts,
                retry_box,
                retry_embedding,
                retry_gender_label,
                retry_gender_conf,
            ):
                first_ts = retry_ts
                first_box = retry_box
                first_embedding = retry_embedding
                first_gender_label = retry_gender_label
                first_gender_conf = retry_gender_conf
                if retry_scan_end is not None:
                    active_scan_end = retry_scan_end
                return True
            return False

        if not has_verified_female_candidate(first_ts, first_box, first_embedding, first_gender_label, first_gender_conf):
            recover_seed_from_retry("no verified seed in initial scan")

        sync_seed_metrics()

        if not has_verified_female_candidate(first_ts, first_box, first_embedding, first_gender_label, first_gender_conf):
            seed_hits = int(scan_info.get("female_seed_hits", 0))
            best_seed_conf = float(scan_info.get("best_seed_confidence", 0.0))
            if seed_hits > 0:
                set_decision(
                    result,
                    "uncertain",
                    f"Female-like candidates found ({seed_hits}) but none verified across confirmation frames.",
                    0.35 + min(0.25, best_seed_conf * 0.3),
                )
                print(
                    f"[UNCERTAIN] {video_name} female candidates found but not verified "
                    f"(seed_hits={seed_hits}, best_seed_conf={best_seed_conf:.3f})",
                    flush=True,
                )
            else:
                set_decision(result, "no_female", "No meaningful female candidate evidence detected.", 0.9)
                print(f"[NO FEMALE] {video_name}", flush=True)
            return result

        while True:
            kept_embeddings, female_score, male_score, gender_votes, stabilize_info = stabilize_identity(
                cap=cap,
                fps=fps,
                total_frames=total_frames,
                scan_duration=active_scan_end,
                first_ts=first_ts,
                first_box=first_box,
                first_embedding=first_embedding,
                initial_gender_label=first_gender_label,
                initial_gender_conf=first_gender_conf,
                cfg=cfg,
                video_name=video_name,
            )
            reason_metrics = _normalize_reason_metrics(result.get("reason_metrics"))
            reason_metrics["total_faces_evaluated"] = int(reason_metrics.get("total_faces_evaluated", 0)) + int(
                stabilize_info.get("total_faces_evaluated", 0)
            )
            reason_metrics["low_face_area_rejections"] = int(reason_metrics.get("low_face_area_rejections", 0)) + int(
                stabilize_info.get("low_face_area_rejections", 0)
            )
            reason_metrics["stable_embeddings"] = len(kept_embeddings)
            reason_metrics["gender_votes"] = int(gender_votes)
            reason_metrics["female_score"] = round(float(female_score), 4)
            reason_metrics["male_score"] = round(float(male_score), 4)
            result["reason_metrics"] = reason_metrics

            if not kept_embeddings:
                if cfg.reprocess_uncertain_only and recover_seed_from_retry("stabilization failed (no stable identity)"):
                    continue
                set_decision(result, "uncertain", "Female candidate detected but identity could not be stabilized.", 0.4)
                print(f"[UNCERTAIN] {video_name} no stable identity", flush=True)
                return result

            if len(kept_embeddings) < cfg.min_stable_embeddings:
                if cfg.reprocess_uncertain_only and recover_seed_from_retry(
                    f"stabilization weak ({len(kept_embeddings)}<{cfg.min_stable_embeddings} stable embeddings)"
                ):
                    continue
                set_decision(
                    result,
                    "uncertain",
                    f"Low stabilization evidence: only {len(kept_embeddings)} stable embeddings.",
                    0.45,
                )
                print(f"[UNCERTAIN] {video_name} only {len(kept_embeddings)} stable embeddings", flush=True)
                return result

            if gender_votes < cfg.min_stabilization_gender_votes:
                if cfg.reprocess_uncertain_only and recover_seed_from_retry(
                    f"stabilization weak ({gender_votes}<{cfg.min_stabilization_gender_votes} gender votes)"
                ):
                    continue
                set_decision(
                    result,
                    "uncertain",
                    f"Low stabilization evidence: only {gender_votes} gender votes.",
                    0.45,
                )
                print(f"[UNCERTAIN] {video_name} only {gender_votes} gender votes during stabilization", flush=True)
                return result

            total_gender_score = female_score + male_score
            female_ratio = 0.0
            if total_gender_score > 0:
                female_ratio = female_score / total_gender_score
                if female_ratio < cfg.min_female_vote_ratio or female_score <= male_score:
                    if cfg.reprocess_uncertain_only and recover_seed_from_retry(
                        "stabilization gender votes conflicted"
                    ):
                        continue
                    stable_embedding = robust_average_embeddings(kept_embeddings)
                    result["embedding"] = stable_embedding.tolist()
                    result["samples_used"] = len(kept_embeddings)
                    set_decision(
                        result,
                        "uncertain",
                        (
                            "Conflicting gender votes during stabilization "
                            f"(female_score={female_score:.3f}, male_score={male_score:.3f})."
                        ),
                        0.5,
                    )
                    print(
                        f"[UNCERTAIN] {video_name} gender votes disagree "
                        f"(female_score={female_score:.3f}, male_score={male_score:.3f}, ratio={female_ratio:.3f})",
                        flush=True,
                    )
                    return result

            stable_embedding = robust_average_embeddings(kept_embeddings)

            result["female_found"] = True
            result["embedding"] = stable_embedding.tolist()
            result["samples_used"] = len(kept_embeddings)
            set_decision(
                result,
                "female_detected",
                "Consistent female detection and stabilized identity track.",
                0.45 + (0.25 * min(1.0, len(kept_embeddings) / max(1, cfg.min_stable_embeddings))) + (0.30 * female_ratio),
            )

            print(
                f"[FEMALE FOUND] {video_name} at {first_ts:.1f}s with {len(kept_embeddings)} stable embeddings "
                f"(female_score={female_score:.3f}, male_score={male_score:.3f}, votes={gender_votes})",
                flush=True,
            )
            return result

    except StopRequestedError:
        result["stopped"] = True
        set_decision(result, "stopped", "Processing stopped by user request.", 0.0)
        print(f"[STOPPED] {video_name}", flush=True)
        return result
    except Exception as exc:
        result["error"] = f"{exc}\n{traceback.format_exc()}"
        set_decision(result, "error", f"Processing failure: {exc}", 0.0)
        return result
    finally:
        cap.release()

def cluster_embeddings(video_results: Sequence[Dict[str, Any]], eps: float, min_samples: int) -> Dict[str, int]:
    vectors: List[np.ndarray] = []
    videos: List[str] = []

    for item in video_results:
        emb = item.get("embedding")
        if emb is None:
            continue
        vec = np.asarray(emb, dtype=np.float32)
        if vec.ndim != 1 or not np.isfinite(vec).all():
            continue
        vec /= np.linalg.norm(vec) + 1e-8
        vectors.append(vec)
        videos.append(item["video"])

    if not vectors:
        return {}

    matrix = np.vstack(vectors)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = dbscan.fit_predict(matrix)

    cluster_vectors: Dict[int, List[np.ndarray]] = {}
    for vec, label in zip(vectors, labels):
        cluster_vectors.setdefault(int(label), []).append(vec)

    cluster_centroids: Dict[int, np.ndarray] = {}
    for label, vecs in cluster_vectors.items():
        centroid = robust_average_embeddings(vecs)
        cluster_centroids[label] = centroid

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
            if cosine_similarity(base, cluster_centroids[other]) >= CFG.cluster_merge_threshold:  # type: ignore[union-attr]
                merged_labels[other] = next_group
        next_group += 1

    ordered_labels = sorted(set(merged_labels[int(label)] for label in labels))
    folder_ids = {old: idx + 1 for idx, old in enumerate(ordered_labels)}
    return {video: folder_ids[merged_labels[int(label)]] for video, label in zip(videos, labels)}


def move_video(src: Path, dst_dir: Path) -> Path:
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    try:
        if src.resolve() == dst.resolve():
            return src
    except Exception:
        pass
    if dst.exists():
        stem = src.stem
        suffix = src.suffix
        counter = 1
        while True:
            candidate = dst_dir / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                dst = candidate
                break
            counter += 1
    shutil.move(str(src), str(dst))
    return dst


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sort videos by female identity using facenet-pytorch on ROCm")
    parser.add_argument("--input-dir", default=".", help="Directory containing source videos")
    parser.add_argument("--output-dir", default=".", help="Directory to place sorted videos")
    parser.add_argument(
        "--profile",
        default=PROFILE_BALANCED,
        help="Quality preset: fast, balanced, or high_accuracy (aliases: high-accuracy, high accuracy).",
    )
    parser.add_argument(
        "--print-runtime-json",
        action="store_true",
        help="Print detected GPU/CPU runtime info as JSON and exit",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive scan of child folders inside the source directory",
    )
    parser.add_argument(
        "--include-generated-folders",
        action="store_true",
        help="Also scan videos inside generated folders such as Female_* and No_Female_Found",
    )
    parser.add_argument(
        "--reprocess-uncertain-only",
        action="store_true",
        help="Reprocess only videos currently in <output-dir>/Uncertain using stricter settings",
    )
    parser.add_argument(
        "--review-mode",
        action="store_true",
        help="Queue review items; uncertain videos stay in Uncertain so they can be reprocessed first",
    )
    parser.add_argument(
        "--review-confidence-threshold",
        type=float,
        default=0.75,
        help="In review mode, queue female_detected results below this confidence and auto-sort the rest",
    )
    parser.add_argument(
        "--review-list-json",
        action="store_true",
        help="Print review queue state JSON for --output-dir and exit",
    )
    parser.add_argument(
        "--duplicates-scan-json",
        action="store_true",
        help="Scan output identity folders and print duplicate groups JSON",
    )
    parser.add_argument(
        "--duplicates-apply-json",
        default="",
        help="Apply duplicate move action from JSON payload (expects {'paths':[...]}).",
    )
    parser.add_argument(
        "--identity-list-json",
        action="store_true",
        help="Print identity folders JSON for --output-dir and exit",
    )
    parser.add_argument(
        "--review-action",
        default="",
        help="Apply one review action: approve_suggested, move_no_female, reassign_existing, reassign_new, skip",
    )
    parser.add_argument("--review-item-id", default="", help="Review item id for --review-action")
    parser.add_argument("--review-target-folder", default="", help="Target folder for reassign review actions")
    parser.add_argument(
        "--identity-action",
        default="",
        help="Apply one identity action: merge, split, lock, unlock",
    )
    parser.add_argument("--identity-source-folder", default="", help="Source identity folder for merge/split")
    parser.add_argument("--identity-target-folder", default="", help="Target identity folder for merge/split")
    parser.add_argument("--identity-folder", default="", help="Identity folder for lock/unlock")
    parser.add_argument(
        "--identity-video-paths-json",
        default="[]",
        help="JSON list of source video paths for identity split action",
    )
    parser.add_argument(
        "--learning-enabled",
        dest="learning_enabled",
        action="store_true",
        default=True,
        help="Enable memory-assisted learning from manual reviews (default: enabled)",
    )
    parser.add_argument(
        "--no-learning-enabled",
        dest="learning_enabled",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--learning-memory-file",
        default="",
        help="Optional path to global learning memory JSON file",
    )
    parser.add_argument(
        "--learning-auto-threshold",
        type=float,
        default=0.82,
        help="Auto-apply memory label when cosine similarity is at least this value",
    )
    parser.add_argument(
        "--learning-suggest-threshold",
        type=float,
        default=0.74,
        help="Mark uncertain with memory suggestion when similarity is at least this value",
    )
    parser.add_argument(
        "--learning-summary-json",
        action="store_true",
        help="Print learning memory summary JSON with adaptive thresholds and correction trends",
    )
    parser.add_argument(
        "--stop-flag-file",
        default="",
        help="Path to a flag file. If the file appears, processing stops gracefully and moves completed work.",
    )
    parser.add_argument(
        "--live-trace",
        action="store_true",
        help="Show live frame preview while also printing trace lines with video/frame/timestamp",
    )
    parser.add_argument("--max-seconds", type=int, default=60, help="Only analyze the first N seconds")
    parser.add_argument("--sample-every-sec", type=float, default=2.0, help="Sparse scan interval in seconds")
    parser.add_argument("--resize-width", type=int, default=960, help="Resize frames to this max width")
    parser.add_argument(
        "--detection-batch-size",
        type=int,
        default=4,
        help="How many sampled frames to batch together for GPU face detection",
    )
    parser.add_argument(
        "--stabilization-seconds",
        type=float,
        default=8.0,
        help="Collect embeddings for this many seconds after first female detection",
    )
    parser.add_argument(
        "--stabilization-sample-sec",
        type=float,
        default=1.0,
        help="Tracking sample interval after first female detection",
    )
    parser.add_argument(
        "--female-confirmation-frames",
        type=int,
        default=2,
        help="Require this many nearby frame confirmations before accepting the first female identity",
    )
    parser.add_argument(
        "--female-confirmation-window-sec",
        type=float,
        default=2.5,
        help="Time window used to confirm the first female identity",
    )
    parser.add_argument(
        "--min-female-gender-confidence",
        type=float,
        default=0.65,
        help="Minimum OpenCV gender confidence required for a female match",
    )
    parser.add_argument(
        "--min-female-vote-ratio",
        type=float,
        default=0.62,
        help="Minimum female confidence ratio (female_score / total_gender_score) across confirmations",
    )
    parser.add_argument(
        "--min-face-area-ratio",
        type=float,
        default=0.018,
        help="Ignore tiny faces below this fraction of the frame area",
    )
    parser.add_argument(
        "--min-stable-embeddings",
        type=int,
        default=3,
        help="Require at least this many same-person embeddings before accepting a video identity",
    )
    parser.add_argument(
        "--min-stabilization-gender-votes",
        type=int,
        default=3,
        help="Require at least this many stabilization gender votes before accepting the video",
    )
    parser.add_argument("--max-faces-per-frame", type=int, default=4, help="Limit faces checked per frame")
    parser.add_argument("--detection-confidence", type=float, default=0.90, help="Minimum MTCNN face confidence")
    parser.add_argument("--same-person-threshold", type=float, default=0.72, help="Cosine similarity threshold")
    parser.add_argument("--duplicate-threshold", type=float, default=0.985, help="Skip near-duplicate embeddings")
    parser.add_argument("--dbscan-eps", type=float, default=0.35, help="DBSCAN eps with cosine distance")
    parser.add_argument("--dbscan-min-samples", type=int, default=1, help="DBSCAN min samples")
    parser.add_argument(
        "--cluster-merge-threshold",
        type=float,
        default=0.78,
        help="Merge DBSCAN clusters when their centroids are at least this cosine similarity",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(1, min(2, (os.cpu_count() or 4) // 2)),
        help="Parallel video workers; keep low to avoid GPU memory contention",
    )
    parser.add_argument(
        "--force-gpu",
        action="store_true",
        help="Force GPU attempt even on likely unsupported mobile Radeon/iGPU devices",
    )
    parser.add_argument(
        "--use-insightface",
        dest="use_insightface",
        action="store_true",
        default=USE_INSIGHTFACE,
        help="Use InsightFace (RetinaFace + ArcFace) for detection and embeddings (default: enabled).",
    )
    parser.add_argument(
        "--no-use-insightface",
        dest="use_insightface",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gender-model-dir",
        default="",
        help="Directory for cached gender model assets; defaults to <output-dir>/.model_cache",
    )
    parser.add_argument("--gender-proto-path", default="", help="Override path for gender_deploy.prototxt")
    parser.add_argument("--gender-model-path", default="", help="Override path for gender_net.caffemodel")
    return parser.parse_args()


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    cli_tokens = sys.argv[1:]
    selected_profile = apply_profile_defaults(args, cli_tokens)
    if bool(args.reprocess_uncertain_only):
        selected_profile = apply_uncertain_reprocess_defaults(args, cli_tokens)
    apply_insightface_defaults(args, cli_tokens)
    output_dir = Path(args.output_dir).expanduser().resolve()

    temp_cfg = Config(
        input_dir=".",
        output_dir=str(output_dir),
        profile=selected_profile,
        use_insightface=bool(args.use_insightface),
        learning_enabled=bool(args.learning_enabled),
        learning_memory_file=str(args.learning_memory_file or ""),
        learning_auto_threshold=float(args.learning_auto_threshold),
        learning_suggest_threshold=float(args.learning_suggest_threshold),
    )
    memory_path = memory_file_path(temp_cfg)

    if args.print_runtime_json:
        print(json.dumps(get_runtime_info()))
        return 0

    if args.learning_summary_json:
        memory = load_memory(memory_path)
        payload = build_learning_summary(
            memory,
            global_auto_threshold=float(args.learning_auto_threshold),
            global_suggest_threshold=float(args.learning_suggest_threshold),
        )
        payload["memory_file"] = str(memory_path)
        print(json.dumps(payload))
        return 0

    if args.duplicates_scan_json:
        payload = scan_duplicates(output_dir)
        print(json.dumps(payload))
        return 0

    if str(args.duplicates_apply_json or "").strip():
        try:
            payload_obj = json.loads(str(args.duplicates_apply_json))
            payload = apply_duplicate_move(
                output_dir,
                payload_obj,
                memory_path=memory_path if bool(args.learning_enabled) else None,
                learning_auto_threshold=float(args.learning_auto_threshold),
                learning_suggest_threshold=float(args.learning_suggest_threshold),
            )
            state = load_review_state(output_dir)
            payload["pending_count"] = len(pending_review_items(state))
            print(json.dumps(payload))
            return 0
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}))
            return 1

    if args.identity_list_json:
        memory = load_memory(memory_path)
        payload = list_identities(output_dir, memory)
        payload["memory_file"] = str(memory_path)
        print(json.dumps(payload))
        return 0

    if args.identity_action:
        try:
            selected_videos: List[str] = []
            if args.identity_video_paths_json.strip():
                parsed = json.loads(args.identity_video_paths_json)
                if not isinstance(parsed, list):
                    raise RuntimeError("--identity-video-paths-json must be a JSON array")
                selected_videos = [str(item) for item in parsed]

            payload = perform_identity_action(
                output_dir=output_dir,
                memory_path=memory_path,
                embedding_cache_file=str(
                    sorting_embedding_cache_path(output_dir, use_insightface=bool(args.use_insightface))
                ),
                action=str(args.identity_action or ""),
                source_folder=str(args.identity_source_folder or ""),
                target_folder=str(args.identity_target_folder or ""),
                folder_name=str(args.identity_folder or ""),
                selected_videos=selected_videos,
            )
            state = load_review_state(output_dir)
            payload["pending_count"] = len(pending_review_items(state))
            print(json.dumps(payload))
            return 0
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}))
            return 1

    if args.review_list_json:
        state = load_review_state(output_dir)
        state["pending_count"] = len(pending_review_items(state))
        print(json.dumps(state))
        return 0

    if args.review_action:
        if not args.review_item_id.strip():
            print(json.dumps({"ok": False, "error": "--review-item-id is required for --review-action"}))
            return 1
        try:
            updated_item = apply_review_action(
                output_dir=output_dir,
                item_id=args.review_item_id.strip(),
                action=args.review_action.strip(),
                target_folder=args.review_target_folder.strip(),
            )
            learning_updated = False
            learning_match: Dict[str, Any] = {}
            if bool(args.learning_enabled):
                memory = load_memory(memory_path)
                learning_updated = update_learning_from_review_item(
                    updated_item,
                    memory,
                    learning_auto_threshold=float(args.learning_auto_threshold),
                    learning_suggest_threshold=float(args.learning_suggest_threshold),
                )
                if learning_updated:
                    save_memory(memory_path, memory)
                    learning_match = {
                        "memory_file": str(memory_path),
                        "feedback_events": int(memory.get("stats", {}).get("total_feedback_events", 0)),
                    }
            state = load_review_state(output_dir)
            print(
                json.dumps(
                    {
                        "ok": True,
                        "item": updated_item,
                        "pending_count": len(pending_review_items(state)),
                        "learning_updated": learning_updated,
                        "learning_info": learning_match,
                    }
                )
            )
            return 0
        except Exception as exc:
            print(json.dumps({"ok": False, "error": str(exc)}))
            return 1

    ensure_dir(output_dir)
    if bool(args.reprocess_uncertain_only):
        input_dir = output_dir / UNCERTAIN_DIRNAME
        ensure_dir(input_dir)
        if bool(args.review_mode):
            print("[INFO] Review mode enabled for uncertain reprocess.", flush=True)
    else:
        input_dir = Path(args.input_dir).expanduser().resolve()
        if not input_dir.is_dir():
            print(f"Input directory not found: {input_dir}")
            return 1

    cfg = Config(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        profile=selected_profile,
        review_mode=args.review_mode,
        review_confidence_threshold=float(args.review_confidence_threshold),
        reprocess_uncertain_only=bool(args.reprocess_uncertain_only),
        learning_enabled=bool(args.learning_enabled),
        learning_memory_file=str(args.learning_memory_file or ""),
        learning_auto_threshold=float(args.learning_auto_threshold),
        learning_suggest_threshold=float(args.learning_suggest_threshold),
        stop_flag_file=args.stop_flag_file,
        recursive=not args.no_recursive,
        include_generated_folders=args.include_generated_folders,
        live_trace=args.live_trace,
        force_gpu=args.force_gpu,
        use_insightface=bool(args.use_insightface),
        max_seconds=args.max_seconds,
        sample_every_sec=args.sample_every_sec,
        resize_width=args.resize_width,
        detection_batch_size=max(1, args.detection_batch_size),
        stabilization_seconds=args.stabilization_seconds,
        stabilization_sample_sec=args.stabilization_sample_sec,
        female_confirmation_frames=max(1, args.female_confirmation_frames),
        female_confirmation_window_sec=args.female_confirmation_window_sec,
        min_female_gender_confidence=args.min_female_gender_confidence,
        min_female_vote_ratio=float(args.min_female_vote_ratio),
        min_face_area_ratio=max(0.0, float(args.min_face_area_ratio)),
        min_stable_embeddings=max(1, args.min_stable_embeddings),
        min_stabilization_gender_votes=max(1, int(args.min_stabilization_gender_votes)),
        max_faces_per_frame=args.max_faces_per_frame,
        detection_confidence=args.detection_confidence,
        same_person_threshold=args.same_person_threshold,
        duplicate_threshold=args.duplicate_threshold,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min_samples,
        cluster_merge_threshold=args.cluster_merge_threshold,
        max_workers=max(1, args.max_workers),
        gender_model_dir=args.gender_model_dir,
        gender_proto_path=args.gender_proto_path,
        gender_model_path=args.gender_model_path,
    )
    cfg.learning_auto_threshold = max(0.0, min(1.0, cfg.learning_auto_threshold))
    cfg.learning_suggest_threshold = max(0.0, min(cfg.learning_auto_threshold, cfg.learning_suggest_threshold))
    cfg.review_confidence_threshold = max(0.0, min(1.0, float(cfg.review_confidence_threshold)))

    proto_path, model_path = resolve_gender_model_paths(cfg)
    cfg.gender_proto_path = str(proto_path)
    cfg.gender_model_path = str(model_path)

    print_gpu_status()

    videos = list_videos(
        cfg.input_dir,
        recursive=cfg.recursive,
        include_generated_folders=cfg.include_generated_folders,
    )
    if not videos:
        print(f"No videos found in {input_dir}")
        close_live_preview()
        return 0

    run_started_at = datetime.now(timezone.utc)
    report_run_id = create_report_run_id(run_started_at)

    print(f"Found {len(videos)} videos")
    print(f"Run id: {report_run_id}")
    print(f"Preset profile: {cfg.profile}")
    print(f"Reprocess uncertain only: {cfg.reprocess_uncertain_only}")
    if cfg.reprocess_uncertain_only:
        print(f"Uncertain bucket: {output_dir / UNCERTAIN_DIRNAME}")
    print(f"Recursive child-folder scan: {cfg.recursive}")
    print(f"Include generated folders: {cfg.include_generated_folders}")
    print(f"Review mode: {cfg.review_mode}")
    if cfg.review_mode:
        print(f"Review confidence threshold: {cfg.review_confidence_threshold:.2f}")
    print(f"Learning enabled: {cfg.learning_enabled}")
    if cfg.learning_enabled:
        print(f"Learning memory file: {memory_file_path(cfg)}")
        print(
            f"Learning thresholds: auto={cfg.learning_auto_threshold:.2f}, "
            f"suggest={cfg.learning_suggest_threshold:.2f}"
        )
        print(f"Embedding model key: {embedding_model_key(cfg.use_insightface)}")
        print(f"Embedding cache file: {sorting_embedding_cache_path(output_dir, use_insightface=cfg.use_insightface)}")
    if str(cfg.stop_flag_file).strip():
        print(f"Graceful stop flag file: {cfg.stop_flag_file}")
    print(f"Live trace enabled: {cfg.live_trace}")
    print(f"Use InsightFace: {cfg.use_insightface}")
    if cfg.use_insightface:
        print(
            "InsightFace tuned thresholds: "
            f"detection_confidence={cfg.detection_confidence}, "
            f"same_person_threshold={cfg.same_person_threshold}, "
            f"duplicate_threshold={cfg.duplicate_threshold}, "
            f"dbscan_eps={cfg.dbscan_eps}, "
            f"cluster_merge_threshold={cfg.cluster_merge_threshold}"
        )
    if cfg.live_trace and cfg.max_workers > 1:
        print("[WARN] Live frame preview is disabled when max-workers > 1; using text trace only.")
    if torch.cuda.is_available() and cfg.max_workers == 1:
        print("[HINT] GPU detected. Increase --max-workers (e.g. 2-4) to keep the GPU busier.")
    print(f"Using workers: {cfg.max_workers}")
    print(
        "Quality settings: "
        f"confirm_frames={cfg.female_confirmation_frames}, "
        f"min_female_vote_ratio={cfg.min_female_vote_ratio}, "
        f"min_face_area_ratio={cfg.min_face_area_ratio}, "
        f"min_stable_embeddings={cfg.min_stable_embeddings}, "
        f"min_stabilization_gender_votes={cfg.min_stabilization_gender_votes}, "
        f"detection_batch_size={cfg.detection_batch_size}, "
        f"dbscan_eps={cfg.dbscan_eps}, "
        f"cluster_merge_threshold={cfg.cluster_merge_threshold}"
    )

    progress_disabled = not sys.stdout.isatty()
    memory_path = memory_file_path(cfg)
    memory = load_memory(memory_path) if cfg.learning_enabled else {}
    results: List[Dict[str, Any]] = []
    stopped_early = False
    stopped_hits = 0
    stop_flag_path = Path(cfg.stop_flag_file).expanduser() if str(cfg.stop_flag_file).strip() else None

    def stop_flag_is_set() -> bool:
        return bool(stop_flag_path and stop_flag_path.exists())

    if cfg.max_workers == 1:
        init_worker(cfg)
        female_hits = 0
        no_female_hits = 0
        error_hits = 0
        progress = tqdm(videos, total=len(videos), desc="Processing videos", disable=progress_disabled)
        for video in progress:
            if stop_flag_is_set():
                stopped_early = True
                print("[STOP] Stop flag detected. Finishing with processed results only.", flush=True)
                break
            try:
                result = process_video(str(video))
                apply_memory_assist(result, memory, cfg)
                apply_explainability_metadata(result, cfg)
                emit_result_json(result)
                results.append(result)
            except Exception as exc:
                result = new_result_record(str(video), ACTIVE_ACCELERATION)
                result["error"] = f"Worker failure: {exc}"
                set_decision(result, "error", f"Worker failure: {exc}", 0.0)
                apply_explainability_metadata(result, cfg)
                emit_result_json(result)
                results.append(result)

            if result.get("stopped"):
                stopped_hits += 1
                stopped_early = True
                print("[STOP] Current video stopped. Finalizing already processed videos.", flush=True)
                break
            elif result.get("error"):
                error_hits += 1
            elif result.get("female_found"):
                female_hits += 1
            else:
                no_female_hits += 1

            emit_progress(len(results), len(videos), female_hits, no_female_hits, error_hits)

            if not progress_disabled:
                progress.set_postfix(
                    {
                        "done": len(results),
                        "female": female_hits,
                        "no_female": no_female_hits,
                        "errors": error_hits,
                    }
                )
    else:
        executor = ProcessPoolExecutor(max_workers=cfg.max_workers, initializer=init_worker, initargs=(cfg,))
        futures: Dict[Any, Path] = {}
        try:
            futures = {executor.submit(process_video, str(video)): video for video in videos}
            female_hits = 0
            no_female_hits = 0
            error_hits = 0
            progress = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Processing videos",
                disable=progress_disabled,
            )
            for future in progress:
                video = futures[future]
                try:
                    result = future.result()
                    apply_memory_assist(result, memory, cfg)
                    apply_explainability_metadata(result, cfg)
                    emit_result_json(result)
                    results.append(result)
                except Exception as exc:
                    result = new_result_record(str(video), "unknown")
                    result["error"] = f"Worker failure: {exc}"
                    set_decision(result, "error", f"Worker failure: {exc}", 0.0)
                    apply_explainability_metadata(result, cfg)
                    emit_result_json(result)
                    results.append(result)

                if result.get("stopped"):
                    stopped_hits += 1
                    stopped_early = True
                elif result.get("error"):
                    error_hits += 1
                elif result.get("female_found"):
                    female_hits += 1
                else:
                    no_female_hits += 1

                emit_progress(len(results), len(videos), female_hits, no_female_hits, error_hits)

                if not progress_disabled:
                    progress.set_postfix(
                        {
                            "done": len(results),
                            "female": female_hits,
                            "no_female": no_female_hits,
                            "errors": error_hits,
                        }
                    )
                if stop_flag_is_set():
                    stopped_early = True
                    print("[STOP] Stop flag detected. Cancelling pending workers and finalizing processed results.", flush=True)
                    break
        finally:
            if stopped_early:
                for future in futures:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)

    for item in results:
        if item.get("error"):
            safe_console_print(f"[WARN] {item['video']}: {item['error']}")

    successful_results = [item for item in results if not item.get("error") and not item.get("stopped")]
    failed_results = [item for item in results if item.get("error")]
    stopped_results = [item for item in results if item.get("stopped")]
    if not successful_results:
        counts = derive_decision_counts(results)
        run_finished_at = datetime.now(timezone.utc)
        emit_run_reports(
            output_dir=output_dir,
            run_id=report_run_id,
            run_started_at=run_started_at,
            run_finished_at=run_finished_at,
            total_scanned=len(videos),
            processed_successfully=counts["processed_successfully"],
            female_detected=counts["female_detected"],
            uncertain=counts["uncertain"],
            no_female_found=counts["no_female_found"],
            errors=counts["errors"],
            stopped_early=stopped_early,
            results=results,
        )
        if stopped_early or stopped_results:
            print("\n[STOP] No completed videos to move yet. Stopped cleanly.")
            close_live_preview()
            return 0
        print("\n[ERROR] No videos were processed successfully, so nothing will be moved.")
        close_live_preview()
        return 1

    cluster_map = cluster_embeddings(results, cfg.dbscan_eps, cfg.dbscan_min_samples)
    cluster_folder_names = build_cluster_folder_names(cluster_map)
    if cluster_folder_names:
        print("\n[CLUSTERS]")
        for cluster_id in sorted(cluster_folder_names.keys()):
            print(f"  cluster {cluster_id} -> {cluster_folder_names[cluster_id]}")

    for item in successful_results:
        if bool(item.get("memory_applied")) and str(item.get("memory_match_label", "")).strip():
            item["suggested_folder_name"] = sanitize_folder_name(str(item.get("memory_match_label", "")).strip())
            item["suggested_cluster_id"] = None
            continue
        cluster_id = cluster_map.get(str(item.get("video", "")))
        if cluster_id is None:
            if item.get("decision_label") == "uncertain":
                item["suggested_folder_name"] = "Needs_Review"
            continue
        cluster_id_int = int(cluster_id)
        item["suggested_cluster_id"] = cluster_id_int
        item["suggested_folder_name"] = cluster_folder_names.get(cluster_id_int, f"Female_{cluster_id_int}")

    female_count = 0
    uncertain_count = 0
    no_female_count = 0
    moved_count = 0
    queued_count = 0
    memory_applied_count = 0
    queue_items: List[Dict[str, Any]] = []
    no_female_dir = output_dir / "No_Female_Found"
    uncertain_dir = output_dir / UNCERTAIN_DIRNAME

    for item in tqdm(results, desc="Sorting videos", disable=progress_disabled):
        if item.get("error") or item.get("stopped"):
            continue
        src = Path(item["video"])
        if not src.exists():
            continue

        decision_label = str(item.get("decision_label", "")).strip().lower()
        route = decide_review_route(
            item,
            cfg.review_mode,
            review_confidence_threshold=cfg.review_confidence_threshold,
        )

        if decision_label == "uncertain":
            uncertain_count += 1
            moved_path = move_video(src, uncertain_dir)
            item["final_destination"] = str(moved_path)
            if moved_path != src:
                moved_count += 1

            # Legacy behavior moved uncertain videos to Review_Pending when queued.
            # New behavior keeps uncertain videos in Uncertain so users can reprocess first,
            # while still allowing optional review queue actions from the same path.
            if cfg.review_mode and route == "queue":
                suggested_folder = str(item.get("suggested_folder_name", "")).strip()
                if not suggested_folder:
                    suggested_folder = "Needs_Review"
                suggested_folder = sanitize_folder_name(suggested_folder)
                queued_count += 1
                queue_items.append(
                    {
                        "id": uuid.uuid4().hex,
                        "source_path": str(item.get("video", "")),
                        "pending_path": str(moved_path),
                        "predicted_label": decision_label or "unknown",
                        "reason": str(item.get("decision_reason", "")),
                        "confidence": float(item.get("confidence_score", 0.0)),
                        "suggested_folder": suggested_folder,
                        "embedding": item.get("embedding"),
                        "memory_suggestion": str(item.get("memory_suggestion", "")),
                        "memory_match_label": str(item.get("memory_match_label", "")),
                        "memory_match_score": float(item.get("memory_match_score", 0.0)),
                        "memory_applied": bool(item.get("memory_applied", False)),
                        "status": "pending",
                        "final_path": "",
                        "reviewed_at": "",
                        "queued_from_uncertain": True,
                    }
                )
            continue

        if route == "queue":
            suggested_folder = str(item.get("suggested_folder_name", "")).strip()
            if not suggested_folder:
                suggested_folder = "Needs_Review" if decision_label == "uncertain" else "Female_Unknown"
            suggested_folder = sanitize_folder_name(suggested_folder)
            pending_dir = output_dir / REVIEW_PENDING_DIRNAME / suggested_folder
            pending_path = move_video(src, pending_dir)
            item["final_destination"] = str(pending_path)
            moved_count += 1
            queued_count += 1
            if decision_label == "female_detected":
                female_count += 1
            elif decision_label == "uncertain":
                uncertain_count += 1
            if bool(item.get("memory_applied")):
                memory_applied_count += 1

            queue_items.append(
                {
                    "id": uuid.uuid4().hex,
                    "source_path": str(item.get("video", "")),
                    "pending_path": str(pending_path),
                    "predicted_label": decision_label or "unknown",
                    "reason": str(item.get("decision_reason", "")),
                    "confidence": float(item.get("confidence_score", 0.0)),
                    "suggested_folder": suggested_folder,
                    "embedding": item.get("embedding"),
                    "memory_suggestion": str(item.get("memory_suggestion", "")),
                    "memory_match_label": str(item.get("memory_match_label", "")),
                    "memory_match_score": float(item.get("memory_match_score", 0.0)),
                    "memory_applied": bool(item.get("memory_applied", False)),
                    "status": "pending",
                    "final_path": "",
                    "reviewed_at": "",
                }
            )
            continue

        preferred_folder = str(item.get("suggested_folder_name", "")).strip()
        if item.get("female_found") and preferred_folder:
            female_count += 1
            dst_dir = output_dir / sanitize_folder_name(preferred_folder)
            if bool(item.get("memory_applied")):
                memory_applied_count += 1
        elif item.get("female_found") and str(src) in cluster_map:
            female_count += 1
            cluster_id = int(cluster_map[str(src)])
            dst_name = cluster_folder_names.get(cluster_id, f"Female_{cluster_id}")
            dst_dir = output_dir / sanitize_folder_name(dst_name)
        else:
            no_female_count += 1
            dst_dir = no_female_dir

        moved_path = move_video(src, dst_dir)
        item["final_destination"] = str(moved_path)
        moved_count += 1

    if cfg.review_mode:
        queue_run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        state = create_review_state(output_dir, queue_items, queue_run_id)
        save_review_state(output_dir, state)
        print(
            f"\n[REVIEW_QUEUE] run_id={queue_run_id} queued={len(queue_items)} "
            f"state={review_state_path(output_dir)}",
            flush=True,
        )

    cache_payload = persist_sorted_embedding_cache(
        output_dir,
        results,
        use_insightface=cfg.use_insightface,
    )
    print(
        f"[LEARNING_CACHE] path={cache_payload['path']} "
        f"updated={cache_payload['updated_entries']} total={cache_payload['total_entries']}",
        flush=True,
    )

    run_finished_at = datetime.now(timezone.utc)
    time_taken_seconds = max(0.0, (run_finished_at - run_started_at).total_seconds())
    emit_run_reports(
        output_dir=output_dir,
        run_id=report_run_id,
        run_started_at=run_started_at,
        run_finished_at=run_finished_at,
        total_scanned=len(videos),
        processed_successfully=len(successful_results),
        female_detected=female_count,
        uncertain=uncertain_count,
        no_female_found=no_female_count,
        errors=len(failed_results),
        stopped_early=stopped_early,
        results=results,
    )

    print("\n===== Summary =====")
    print(f"Total videos: {len(videos)}")
    print(f"Processed successfully: {len(successful_results)}")
    print(f"Processing errors: {len(failed_results)}")
    print(f"Stopped during processing: {len(stopped_results)}")
    print(f"Stopped early flag: {stopped_early}")
    print(f"Female detected: {female_count}")
    print(f"Uncertain: {uncertain_count}")
    print(f"No female found: {no_female_count}")
    print(f"Queued for review: {queued_count}")
    print(f"Memory auto-applied: {memory_applied_count}")
    print(f"Moved videos: {moved_count}")
    print(f"Videos left in source: {max(0, len(videos) - moved_count)}")
    if cfg.reprocess_uncertain_only:
        print(
            f"[UNCERTAIN_RERUN] processed={len(successful_results)} "
            f"reclassified={female_count + no_female_count} remaining_uncertain={uncertain_count}"
        )
    print(f"Time taken (sec): {time_taken_seconds:.3f}")
    print(f"Output directory: {output_dir}")
    print("===================")
    close_live_preview()
    return 0


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    sys.exit(main())
