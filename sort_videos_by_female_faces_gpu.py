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
4. Uses MTCNN on GPU for face detection.
5. Uses InceptionResnetV1 on GPU for face embeddings.
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
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import traceback
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image
from sklearn.cluster import DBSCAN
from tqdm import tqdm


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


@dataclass
class Config:
    input_dir: str
    output_dir: str
    recursive: bool = True
    include_generated_folders: bool = False
    live_trace: bool = False
    force_gpu: bool = False
    max_seconds: int = 60
    sample_every_sec: float = 2.0
    resize_width: int = 960
    detection_batch_size: int = 4
    stabilization_seconds: float = 8.0
    stabilization_sample_sec: float = 1.0
    female_confirmation_frames: int = 2
    female_confirmation_window_sec: float = 2.5
    min_female_gender_confidence: float = 0.65
    min_stable_embeddings: int = 3
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


def init_worker(cfg: Config) -> None:
    global CFG, DEVICE, MTCNN_MODEL, EMBED_MODEL, GENDER_NET, ACTIVE_ACCELERATION
    CFG = cfg

    proto_path, model_path = resolve_gender_model_paths(cfg)

    GENDER_NET = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
    GENDER_NET.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    GENDER_NET.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    preferred_device, preferred_reason = choose_preferred_device(force_gpu=cfg.force_gpu)
    print(f"[INFO] Preferred acceleration: {preferred_device.type} | {preferred_reason}", flush=True)

    def _load_models(device: torch.device) -> None:
        global DEVICE, MTCNN_MODEL, EMBED_MODEL, ACTIVE_ACCELERATION
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
        ACTIVE_ACCELERATION = "GPU via ROCm" if device.type == "cuda" else "CPU fallback"

    try:
        _load_models(preferred_device)
    except Exception as exc:
        if preferred_device.type == "cuda" and should_fallback_to_cpu(exc):
            print(f"[WARN] GPU initialization failed, falling back to CPU: {exc}", flush=True)
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            _load_models(torch.device("cpu"))
        else:
            raise


def list_videos(input_dir: str, recursive: bool = True, include_generated_folders: bool = False) -> List[Path]:
    root = Path(input_dir)
    iterator = root.rglob("*") if recursive else root.glob("*")
    videos: List[Path] = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTS:
            continue
        rel_parts = path.relative_to(root).parts
        rel_parts_set = set(rel_parts)
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


def detect_faces(frame_rgb: np.ndarray) -> List[Dict[str, Any]]:
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
            }
        )

    detections.sort(key=lambda item: (item["area"] * item["prob"]), reverse=True)
    return detections


def _detect_faces_single(frame_rgb: np.ndarray) -> List[Dict[str, Any]]:
    assert MTCNN_MODEL is not None
    boxes, probs = MTCNN_MODEL.detect(Image.fromarray(frame_rgb))
    return _format_detections_for_frame(boxes, probs, frame_rgb)


def detect_faces_batch(frames_rgb: Sequence[np.ndarray]) -> List[List[Dict[str, Any]]]:
    assert MTCNN_MODEL is not None
    if not frames_rgb:
        return []

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


def embed_faces(face_crops_rgb: Sequence[np.ndarray]) -> np.ndarray:
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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)


def build_sample_times(max_seconds: float, step_seconds: float) -> List[float]:
    times: List[float] = []
    t = 0.0
    while t <= max_seconds:
        times.append(t)
        t += step_seconds
    return times


def emit_trace(video_name: str, phase: str, timestamp_sec: float, frame_index: int) -> None:
    if CFG is None or not CFG.live_trace:
        return
    print(
        f"[TRACE] video={video_name} phase={phase} time={timestamp_sec:.2f}s frame={frame_index}",
        flush=True,
    )


def emit_progress(done: int, total: int, female: int, no_female: int, errors: int) -> None:
    print(
        f"[PROGRESS] done={done} total={total} female={female} no_female={no_female} errors={errors}",
        flush=True,
    )


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
) -> bool:
    confirmations = 1
    step = max(0.5, cfg.female_confirmation_window_sec / max(1, cfg.female_confirmation_frames))
    for idx in range(1, cfg.female_confirmation_frames):
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
        gender_label, gender_conf = classify_gender(face_bgr)
        if gender_label.lower() == "female" and gender_conf >= cfg.min_female_gender_confidence:
            confirmations += 1

    return confirmations >= cfg.female_confirmation_frames


def find_first_female(
    cap: cv2.VideoCapture,
    fps: float,
    total_frames: int,
    scan_duration: float,
    cfg: Config,
    video_name: str,
) -> Tuple[Optional[float], Optional[Tuple[int, int, int, int]], Optional[np.ndarray]]:
    sample_times = build_sample_times(scan_duration, cfg.sample_every_sec)
    sample_indices = [timestamp_to_frame_index(ts, fps, total_frames) for ts in sample_times]

    for start in range(0, len(sample_times), cfg.detection_batch_size):
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

                expanded = expand_box(face["bbox"], cfg.mtcnn_margin_px, frame_bgr.shape[1], frame_bgr.shape[0])
                face_bgr = crop_bgr(frame_bgr, expanded)
                if face_bgr is None:
                    continue

                gender_label, gender_conf = classify_gender(face_bgr)
                if gender_label.lower() != "female" or gender_conf < cfg.min_female_gender_confidence:
                    continue

                face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
                emb_batch = embed_faces([face_rgb])
                if len(emb_batch) == 0:
                    continue

                if not verify_female_candidate(
                    cap=cap,
                    fps=fps,
                    total_frames=total_frames,
                    scan_duration=scan_duration,
                    first_ts=timestamp_sec,
                    first_embedding=emb_batch[0],
                    cfg=cfg,
                ):
                    continue

                print(f"[SCAN] first female at {timestamp_sec:.1f}s (gender_conf={gender_conf:.3f})", flush=True)
                return timestamp_sec, expanded, emb_batch[0]

    return None, None, None


def select_best_matching_face(
    frame_bgr: np.ndarray,
    faces: List[Dict[str, Any]],
    reference_embedding: np.ndarray,
    cfg: Config,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]], float]:
    candidate_boxes: List[Tuple[int, int, int, int]] = []
    candidate_crops_rgb: List[np.ndarray] = []

    for face in faces[: cfg.max_faces_per_frame]:
        if face["prob"] < cfg.detection_confidence:
            continue
        expanded = expand_box(face["bbox"], cfg.mtcnn_margin_px, frame_bgr.shape[1], frame_bgr.shape[0])
        crop = crop_bgr(frame_bgr, expanded)
        if crop is None:
            continue
        candidate_boxes.append(expanded)
        candidate_crops_rgb.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    if not candidate_crops_rgb:
        return None, None, -1.0

    embeddings = embed_faces(candidate_crops_rgb)
    if len(embeddings) == 0:
        return None, None, -1.0

    sims = [cosine_similarity(reference_embedding, emb) for emb in embeddings]
    best_idx = int(np.argmax(sims))
    return embeddings[best_idx], candidate_boxes[best_idx], float(sims[best_idx])


def stabilize_identity(
    cap: cv2.VideoCapture,
    fps: float,
    total_frames: int,
    scan_duration: float,
    first_ts: float,
    first_box: Tuple[int, int, int, int],
    first_embedding: np.ndarray,
    cfg: Config,
    video_name: str,
) -> List[np.ndarray]:
    end_ts = min(scan_duration, first_ts + cfg.stabilization_seconds)
    if end_ts <= first_ts:
        return [first_embedding]

    kept_embeddings: List[np.ndarray] = [first_embedding]
    running_reference = first_embedding.copy()
    last_kept = first_embedding.copy()
    current_box = first_box

    sample_offsets = build_sample_times(end_ts - first_ts, cfg.stabilization_sample_sec)[1:]
    sample_times = [first_ts + offset_sec for offset_sec in sample_offsets]
    sample_indices = [timestamp_to_frame_index(ts, fps, total_frames) for ts in sample_times]

    for start in range(0, len(sample_times), cfg.detection_batch_size):
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

            kept_embeddings.append(matched_embedding)
            last_kept = matched_embedding
            current_box = matched_box
            running_reference = np.mean(np.vstack(kept_embeddings), axis=0).astype(np.float32)
            running_reference /= np.linalg.norm(running_reference) + 1e-8

    return kept_embeddings


def process_video(video_path: str) -> Dict[str, Any]:
    assert CFG is not None
    cfg = CFG
    video_name = Path(video_path).name

    print(f"[START] {video_name}", flush=True)

    result: Dict[str, Any] = {
        "video": video_path,
        "female_found": False,
        "embedding": None,
        "error": None,
        "samples_used": 0,
        "device": ACTIVE_ACCELERATION,
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        result["error"] = "Failed to open video"
        return result

    try:
        fps, total_frames, duration_sec = get_video_meta(cap)
        if total_frames <= 0:
            result["error"] = "Invalid or empty video"
            return result

        scan_duration = min(float(cfg.max_seconds), duration_sec if duration_sec > 0 else float(cfg.max_seconds))
        first_ts, first_box, first_embedding = find_first_female(
            cap,
            fps,
            total_frames,
            scan_duration,
            cfg,
            video_name,
        )
        if first_ts is None or first_box is None or first_embedding is None:
            print(f"[NO FEMALE] {video_name}", flush=True)
            return result

        kept_embeddings = stabilize_identity(
            cap=cap,
            fps=fps,
            total_frames=total_frames,
            scan_duration=scan_duration,
            first_ts=first_ts,
            first_box=first_box,
            first_embedding=first_embedding,
            cfg=cfg,
            video_name=video_name,
        )
        if not kept_embeddings:
            print(f"[NO STABLE ID] {video_name}", flush=True)
            return result

        if len(kept_embeddings) < cfg.min_stable_embeddings:
            print(f"[LOW CONFIDENCE] {video_name} only {len(kept_embeddings)} stable embeddings", flush=True)
            return result

        stable_embedding = robust_average_embeddings(kept_embeddings)

        result["female_found"] = True
        result["embedding"] = stable_embedding.tolist()
        result["samples_used"] = len(kept_embeddings)

        print(
            f"[FEMALE FOUND] {video_name} at {first_ts:.1f}s with {len(kept_embeddings)} stable embeddings",
            flush=True,
        )
        return result

    except Exception as exc:
        result["error"] = f"{exc}\n{traceback.format_exc()}"
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
    parser.add_argument("--input-dir", required=True, help="Directory containing source videos")
    parser.add_argument("--output-dir", required=True, help="Directory to place sorted videos")
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
        "--live-trace",
        action="store_true",
        help="Print live trace lines showing the current video and frame/timestamp being checked",
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
        "--min-stable-embeddings",
        type=int,
        default=3,
        help="Require at least this many same-person embeddings before accepting a video identity",
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
        "--gender-model-dir",
        default="",
        help="Directory for cached gender model assets; defaults to <output-dir>/.model_cache",
    )
    parser.add_argument("--gender-proto-path", default="", help="Override path for gender_deploy.prototxt")
    parser.add_argument("--gender-model-path", default="", help="Override path for gender_net.caffemodel")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.print_runtime_json:
        print(json.dumps(get_runtime_info()))
        return 0

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}")
        return 1

    ensure_dir(output_dir)

    cfg = Config(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        recursive=not args.no_recursive,
        include_generated_folders=args.include_generated_folders,
        live_trace=args.live_trace,
        force_gpu=args.force_gpu,
        max_seconds=args.max_seconds,
        sample_every_sec=args.sample_every_sec,
        resize_width=args.resize_width,
        detection_batch_size=max(1, args.detection_batch_size),
        stabilization_seconds=args.stabilization_seconds,
        stabilization_sample_sec=args.stabilization_sample_sec,
        female_confirmation_frames=max(1, args.female_confirmation_frames),
        female_confirmation_window_sec=args.female_confirmation_window_sec,
        min_female_gender_confidence=args.min_female_gender_confidence,
        min_stable_embeddings=max(1, args.min_stable_embeddings),
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
        return 0

    print(f"Found {len(videos)} videos")
    print(f"Recursive child-folder scan: {cfg.recursive}")
    print(f"Include generated folders: {cfg.include_generated_folders}")
    print(f"Live trace enabled: {cfg.live_trace}")
    print(f"Using workers: {cfg.max_workers}")
    print(
        "Quality settings: "
        f"confirm_frames={cfg.female_confirmation_frames}, "
        f"min_stable_embeddings={cfg.min_stable_embeddings}, "
        f"detection_batch_size={cfg.detection_batch_size}, "
        f"dbscan_eps={cfg.dbscan_eps}, "
        f"cluster_merge_threshold={cfg.cluster_merge_threshold}"
    )

    progress_disabled = not sys.stdout.isatty()
    results: List[Dict[str, Any]] = []
    if cfg.max_workers == 1:
        init_worker(cfg)
        female_hits = 0
        no_female_hits = 0
        error_hits = 0
        progress = tqdm(videos, total=len(videos), desc="Processing videos", disable=progress_disabled)
        for video in progress:
            try:
                result = process_video(str(video))
                results.append(result)
            except Exception as exc:
                result = {
                    "video": str(video),
                    "female_found": False,
                    "embedding": None,
                    "error": f"Worker failure: {exc}",
                    "samples_used": 0,
                    "device": ACTIVE_ACCELERATION,
                }
                results.append(result)

            if result.get("error"):
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
        with ProcessPoolExecutor(max_workers=cfg.max_workers, initializer=init_worker, initargs=(cfg,)) as executor:
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
                    results.append(result)
                except Exception as exc:
                    result = {
                        "video": str(video),
                        "female_found": False,
                        "embedding": None,
                        "error": f"Worker failure: {exc}",
                        "samples_used": 0,
                        "device": "unknown",
                    }
                    results.append(result)

                if result.get("error"):
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

    for item in results:
        if item.get("error"):
            print(f"[WARN] {item['video']}: {item['error']}")

    successful_results = [item for item in results if not item.get("error")]
    failed_results = [item for item in results if item.get("error")]
    if not successful_results:
        print("\n[ERROR] No videos were processed successfully, so nothing will be moved.")
        return 1

    cluster_map = cluster_embeddings(results, cfg.dbscan_eps, cfg.dbscan_min_samples)

    female_count = 0
    moved_count = 0
    no_female_dir = output_dir / "No_Female_Found"

    for item in tqdm(results, desc="Sorting videos", disable=progress_disabled):
        if item.get("error"):
            continue
        src = Path(item["video"])
        if not src.exists():
            continue

        if item.get("female_found") and str(src) in cluster_map:
            female_count += 1
            dst_dir = output_dir / f"Female_{cluster_map[str(src)]}"
        else:
            dst_dir = no_female_dir

        move_video(src, dst_dir)
        moved_count += 1

    print("\n===== Summary =====")
    print(f"Total videos: {len(videos)}")
    print(f"Processed successfully: {len(successful_results)}")
    print(f"Processing errors: {len(failed_results)}")
    print(f"Female detected: {female_count}")
    print(f"No female found: {len(successful_results) - female_count}")
    print(f"Moved videos: {moved_count}")
    print(f"Output directory: {output_dir}")
    print("===================")
    return 0


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    sys.exit(main())
