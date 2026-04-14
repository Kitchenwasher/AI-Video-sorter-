from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from learning_memory import load_memory, record_structural_feedback, save_memory


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
EXCLUDED_DIRS = {
    "No_Female_Found",
    "Uncertain",
    "Review_Pending",
    ".review_queue",
    ".learning",
    ".model_cache",
    "Duplicates",
    "__pycache__",
}


def _iter_identity_videos(output_dir: Path) -> List[Path]:
    if not output_dir.is_dir():
        return []
    videos: List[Path] = []
    for child in sorted(output_dir.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        if child.name in EXCLUDED_DIRS or child.name.startswith("."):
            continue
        for item in sorted(child.iterdir(), key=lambda p: p.name.lower()):
            if item.is_file() and item.suffix.lower() in VIDEO_EXTS:
                videos.append(item.resolve())
    return videos


def _file_hash_sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _open_video_meta(path: Path) -> Dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    width = 0
    height = 0
    fps = 0.0
    frame_count = 0
    duration = 0.0
    try:
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps > 0 and frame_count > 0:
                duration = float(frame_count) / float(fps)
    finally:
        cap.release()
    return {
        "width": max(0, width),
        "height": max(0, height),
        "fps": max(0.0, fps),
        "frame_count": max(0, frame_count),
        "duration": max(0.0, duration),
    }


def _phash_frame(frame_bgr: np.ndarray) -> int:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    block = dct[:8, :8]
    flattened = block.flatten()
    median = float(np.median(flattened[1:])) if flattened.size > 1 else 0.0
    bits = (flattened > median).astype(np.uint8)
    # Vectorized bit-packing: ~10x faster than Python loop over 64 bits.
    packed = np.packbits(bits)
    value = int.from_bytes(packed.tobytes(), byteorder="big")
    return value


def _sample_video_phash(path: Path, sample_count: int = 6) -> List[int]:
    cap = cv2.VideoCapture(str(path))
    hashes: List[int] = []
    try:
        if not cap.isOpened():
            return hashes
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames <= 0:
            return hashes

        sample_count = max(3, int(sample_count))
        points = np.linspace(0.08, 0.92, num=sample_count)
        indices = []
        for frac in points:
            idx = int(round(frac * max(0, total_frames - 1)))
            indices.append(max(0, min(total_frames - 1, idx)))

        seen = set()
        for idx in indices:
            if idx in seen:
                continue
            seen.add(idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            hashes.append(_phash_frame(frame))
    finally:
        cap.release()
    return hashes


def _hamming_distance(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def _average_hash_distance(a_hashes: Sequence[int], b_hashes: Sequence[int]) -> float:
    if not a_hashes or not b_hashes:
        return 999.0
    n = min(len(a_hashes), len(b_hashes))
    if n <= 0:
        return 999.0
    distances = [_hamming_distance(int(a_hashes[i]), int(b_hashes[i])) for i in range(n)]
    return float(sum(distances)) / float(len(distances))


def _duration_close(a: float, b: float, max_seconds: float = 1.0, max_ratio: float = 0.08) -> bool:
    if a <= 0 or b <= 0:
        return False
    diff = abs(a - b)
    allowed = max(max_seconds, max(a, b) * max_ratio)
    return diff <= allowed


def _choose_primary(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ranked = sorted(
        items,
        key=lambda row: (
            -(int(row.get("width", 0)) * int(row.get("height", 0))),
            -float(row.get("estimated_bitrate", 0.0)),
            -int(row.get("size", 0)),
            str(row.get("path", "")),
        ),
    )
    return ranked[0]


def _build_item(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    size = int(stat.st_size)
    meta = _open_video_meta(path)
    duration = float(meta.get("duration", 0.0))
    estimated_bitrate = float(size) / duration if duration > 0 else 0.0
    return {
        "path": str(path),
        "folder": path.parent.name,
        "size": size,
        "hash_sha1": _file_hash_sha1(path),
        "duration": duration,
        "width": int(meta.get("width", 0)),
        "height": int(meta.get("height", 0)),
        "estimated_bitrate": estimated_bitrate,
        "phash_samples": _sample_video_phash(path),
    }


def _build_exact_groups(items: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    by_exact: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for item in items:
        key = (int(item.get("size", 0)), str(item.get("hash_sha1", "")))
        by_exact.setdefault(key, []).append(item)
    groups = [group for group in by_exact.values() if len(group) > 1]
    groups.sort(key=lambda group: str(group[0].get("path", "")))
    return groups


def _build_near_groups(
    items: Sequence[Dict[str, Any]],
    used_paths: set[str],
    distance_threshold: float = 8.0,
) -> List[List[Dict[str, Any]]]:
    candidates = [row for row in items if str(row.get("path", "")) not in used_paths]
    if len(candidates) < 2:
        return []

    n = len(candidates)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Pre-bucket by rounded duration to avoid O(n²) brute-force.
    # Only compare candidates whose durations are close (bucket overlap).
    bucket_width = 2.0  # seconds
    duration_buckets: Dict[int, List[int]] = {}
    for idx, item in enumerate(candidates):
        dur = float(item.get("duration", 0.0))
        bucket_key = int(dur / bucket_width)
        for bk in (bucket_key - 1, bucket_key, bucket_key + 1):
            duration_buckets.setdefault(bk, []).append(idx)

    compared: set[Tuple[int, int]] = set()
    for bucket_indices in duration_buckets.values():
        for bi in range(len(bucket_indices)):
            i = bucket_indices[bi]
            item_i = candidates[i]
            for bj in range(bi + 1, len(bucket_indices)):
                j = bucket_indices[bj]
                pair = (min(i, j), max(i, j))
                if pair in compared:
                    continue
                compared.add(pair)
                item_j = candidates[j]
                if not _duration_close(float(item_i.get("duration", 0.0)), float(item_j.get("duration", 0.0))):
                    continue
                distance = _average_hash_distance(
                    item_i.get("phash_samples", []),
                    item_j.get("phash_samples", []),
                )
                if distance <= distance_threshold:
                    union(i, j)

    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for idx, item in enumerate(candidates):
        grouped.setdefault(find(idx), []).append(item)

    near_groups = [group for group in grouped.values() if len(group) > 1]
    near_groups.sort(key=lambda group: str(group[0].get("path", "")))
    return near_groups


def scan_duplicates(output_dir: Path) -> Dict[str, Any]:
    videos = _iter_identity_videos(output_dir)
    items = [_build_item(path) for path in videos]

    groups: List[Dict[str, Any]] = []
    used_paths: set[str] = set()

    exact_groups = _build_exact_groups(items)
    for idx, group in enumerate(exact_groups, start=1):
        primary = _choose_primary(group)
        duplicates = [row for row in group if str(row.get("path", "")) != str(primary.get("path", ""))]
        used_paths.update(str(row.get("path", "")) for row in group)
        groups.append(
            {
                "group_id": f"exact-{idx}",
                "match_type": "exact",
                "primary_path": str(primary.get("path", "")),
                "duplicate_paths": [str(row.get("path", "")) for row in duplicates],
                "scores": {
                    "primary_quality": {
                        "resolution": int(primary.get("width", 0)) * int(primary.get("height", 0)),
                        "estimated_bitrate": float(primary.get("estimated_bitrate", 0.0)),
                        "size": int(primary.get("size", 0)),
                    },
                    "distance_to_primary": [
                        {
                            "path": str(row.get("path", "")),
                            "distance": 0.0,
                        }
                        for row in duplicates
                    ],
                },
            }
        )

    near_groups = _build_near_groups(items, used_paths)
    for idx, group in enumerate(near_groups, start=1):
        primary = _choose_primary(group)
        duplicates = [row for row in group if str(row.get("path", "")) != str(primary.get("path", ""))]
        primary_hashes = primary.get("phash_samples", [])
        groups.append(
            {
                "group_id": f"near-{idx}",
                "match_type": "near",
                "primary_path": str(primary.get("path", "")),
                "duplicate_paths": [str(row.get("path", "")) for row in duplicates],
                "scores": {
                    "primary_quality": {
                        "resolution": int(primary.get("width", 0)) * int(primary.get("height", 0)),
                        "estimated_bitrate": float(primary.get("estimated_bitrate", 0.0)),
                        "size": int(primary.get("size", 0)),
                    },
                    "distance_to_primary": [
                        {
                            "path": str(row.get("path", "")),
                            "distance": round(
                                _average_hash_distance(primary_hashes, row.get("phash_samples", [])),
                                3,
                            ),
                        }
                        for row in duplicates
                    ],
                },
            }
        )

    for group in groups:
        group["duplicate_count"] = len(group.get("duplicate_paths", []))

    return {
        "ok": True,
        "count": len(groups),
        "groups": groups,
        "output_dir": str(output_dir),
    }


def _move_collision_safe(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
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


def apply_duplicate_move(
    output_dir: Path,
    request_payload: Dict[str, Any],
    *,
    memory_path: Optional[Path] = None,
    learning_auto_threshold: float = 0.82,
    learning_suggest_threshold: float = 0.74,
) -> Dict[str, Any]:
    if not isinstance(request_payload, dict):
        raise RuntimeError("duplicates apply payload must be a JSON object")

    raw_paths = request_payload.get("paths", [])
    if not isinstance(raw_paths, list):
        raise RuntimeError("'paths' must be a JSON array")

    duplicates_root = output_dir / "Duplicates"
    moved: List[Dict[str, str]] = []
    skipped: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []
    memory: Optional[Dict[str, Any]] = None
    memory_dirty = False
    if memory_path is not None:
        try:
            memory = load_memory(memory_path)
        except Exception as exc:
            memory = None
            errors.append({"path": "", "error": f"learning memory load failed: {exc}"})

    output_root = output_dir.resolve()
    for raw in raw_paths:
        src_path = Path(str(raw)).expanduser()
        try:
            resolved = src_path.resolve()
        except Exception:
            skipped.append({"path": str(src_path), "reason": "invalid_path"})
            continue

        if not resolved.exists():
            skipped.append({"path": str(resolved), "reason": "missing"})
            continue
        if not resolved.is_file() or resolved.suffix.lower() not in VIDEO_EXTS:
            skipped.append({"path": str(resolved), "reason": "not_video"})
            continue

        try:
            relative = resolved.relative_to(output_root)
        except Exception:
            skipped.append({"path": str(resolved), "reason": "outside_output_dir"})
            continue

        if any(part in EXCLUDED_DIRS for part in relative.parts):
            skipped.append({"path": str(resolved), "reason": "excluded_folder"})
            continue

        source_folder = relative.parts[0] if relative.parts else "Unknown"
        dst_dir = duplicates_root / source_folder
        try:
            moved_to = _move_collision_safe(resolved, dst_dir)
            moved.append({"source_path": str(resolved), "destination_path": str(moved_to.resolve())})
            if memory is not None:
                record_structural_feedback(
                    memory,
                    action="duplicates_move",
                    source_action="duplicates_move",
                    from_label=source_folder,
                    to_label=source_folder,
                    source_path=str(resolved),
                    final_path=str(moved_to.resolve()),
                    global_auto_threshold=learning_auto_threshold,
                    global_suggest_threshold=learning_suggest_threshold,
                )
                memory_dirty = True
        except Exception as exc:
            errors.append({"path": str(resolved), "error": str(exc)})

    if memory is not None and memory_dirty and memory_path is not None:
        try:
            save_memory(memory_path, memory)
        except Exception as exc:
            errors.append({"path": "", "error": f"learning memory save failed: {exc}"})

    return {
        "ok": True,
        "details": {
            "duplicates_dir": str(duplicates_root),
            "moved": moved,
            "skipped": skipped,
            "errors": errors,
            "moved_count": len(moved),
            "skipped_count": len(skipped),
            "error_count": len(errors),
        },
    }
