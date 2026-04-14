from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from folder_naming import sanitize_folder_name
from learning_memory import (
    load_memory,
    normalize_embedding,
    record_feedback,
    record_structural_feedback,
    refresh_all_identity_stats,
    save_memory,
)


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
EXCLUDED_DIRS = {
    "No_Female_Found",
    "Uncertain",
    "Review_Pending",
    ".review_queue",
    ".learning",
    ".model_cache",
    "__pycache__",
}
EMBEDDING_CACHE_DIRNAME = ".learning"
EMBEDDING_CACHE_FILENAME = "video_embedding_cache.json"
EMBEDDING_CACHE_SCHEMA_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def _embedding_cache_path(output_dir: Path, embedding_cache_file: str = "") -> Path:
    explicit = str(embedding_cache_file or "").strip()
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_absolute():
            return path
        return (output_dir / path).resolve()
    return output_dir / EMBEDDING_CACHE_DIRNAME / EMBEDDING_CACHE_FILENAME


def _normalize_cache_key(path_text: str) -> str:
    return os.path.normcase(os.path.normpath(str(path_text).strip()))


def _load_embedding_cache(output_dir: Path, embedding_cache_file: str = "") -> Dict[str, Any]:
    path = _embedding_cache_path(output_dir, embedding_cache_file)
    if not path.exists():
        return {"schema_version": EMBEDDING_CACHE_SCHEMA_VERSION, "updated_at": "", "entries": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"schema_version": EMBEDDING_CACHE_SCHEMA_VERSION, "updated_at": "", "entries": {}}
    if not isinstance(data, dict):
        return {"schema_version": EMBEDDING_CACHE_SCHEMA_VERSION, "updated_at": "", "entries": {}}
    entries = data.get("entries", {})
    if not isinstance(entries, dict):
        entries = {}
    data["schema_version"] = EMBEDDING_CACHE_SCHEMA_VERSION
    data["entries"] = entries
    data.setdefault("updated_at", "")
    return data


def _save_embedding_cache(output_dir: Path, cache: Dict[str, Any], embedding_cache_file: str = "") -> None:
    path = _embedding_cache_path(output_dir, embedding_cache_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache["schema_version"] = EMBEDDING_CACHE_SCHEMA_VERSION
    cache["updated_at"] = utc_now_iso()
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(cache, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _apply_learning_feedback_from_cache_entry(
    memory: Dict[str, Any],
    *,
    source_action: str,
    from_label: str,
    to_label: str,
    source_path: str,
    final_path: str,
    cache_entry: Optional[Dict[str, Any]],
) -> bool:
    if not isinstance(cache_entry, dict):
        return False
    embedding = cache_entry.get("embedding")
    embedding_vec = normalize_embedding(embedding if isinstance(embedding, list) else [])
    reid_embedding = cache_entry.get("reid_embedding")
    reid_embedding_vec = normalize_embedding(reid_embedding if isinstance(reid_embedding, list) else [])
    if embedding_vec is None and reid_embedding_vec is None:
        return False

    confidence = _safe_float(cache_entry.get("confidence_score", 0.0), 0.0)
    predicted_label = from_label or str(cache_entry.get("decision_label", "")).strip()
    final_label = to_label

    record_feedback(
        memory,
        action=source_action,
        source_action=source_action,
        feedback_event_type="positive",
        label=final_label,
        predicted_label=predicted_label,
        confidence=confidence,
        source_path=source_path,
        final_path=final_path,
        embedding=embedding_vec.tolist() if embedding_vec is not None else None,
        reid_embedding=reid_embedding_vec.tolist() if reid_embedding_vec is not None else None,
        from_label=from_label,
        to_label=to_label,
    )
    if from_label.strip().lower() != to_label.strip().lower():
        record_feedback(
            memory,
            action=source_action,
            source_action=source_action,
            feedback_event_type="negative",
            label=final_label,
            predicted_label=predicted_label,
            confidence=confidence,
            source_path=source_path,
            final_path=final_path,
            embedding=None,
            from_label=from_label,
            to_label=to_label,
            negative_label=from_label,
        )
    return True


def _normalize_identity_record(identity: Dict[str, Any], label: str) -> Dict[str, Any]:
    normalized = dict(identity)
    normalized["label"] = str(normalized.get("label", label)).strip() or label
    normalized["prototype"] = normalized.get("prototype", [])
    normalized["sample_count"] = int(normalized.get("sample_count", 0) or 0)
    normalized["confidence_sum"] = float(normalized.get("confidence_sum", 0.0) or 0.0)
    normalized["last_used"] = str(normalized.get("last_used", "") or "")
    normalized["locked"] = bool(normalized.get("locked", False))
    normalized["locked_at"] = str(normalized.get("locked_at", "") or "")
    normalized["positive_feedback_count"] = int(normalized.get("positive_feedback_count", 0) or 0)
    normalized["negative_feedback_count"] = int(normalized.get("negative_feedback_count", 0) or 0)
    normalized["correction_consistency_score"] = float(normalized.get("correction_consistency_score", 0.0) or 0.0)
    normalized["adaptive_auto_threshold"] = float(normalized.get("adaptive_auto_threshold", 0.82) or 0.82)
    normalized["last_corrected_at"] = str(normalized.get("last_corrected_at", "") or "")
    reid_proto = normalized.get("reid_prototype", [])
    normalized["reid_prototype"] = reid_proto if isinstance(reid_proto, list) else []
    normalized["reid_sample_count"] = int(normalized.get("reid_sample_count", 0) or 0)
    normalized["reid_last_used"] = str(normalized.get("reid_last_used", "") or "")
    reid_same_person_threshold = normalized.get("reid_same_person_threshold", None)
    try:
        normalized["reid_same_person_threshold"] = (
            None if reid_same_person_threshold is None else float(reid_same_person_threshold)
        )
    except Exception:
        normalized["reid_same_person_threshold"] = None
    return normalized


def _iter_identity_folders(output_dir: Path) -> Iterable[Path]:
    if not output_dir.is_dir():
        return []
    folders: List[Path] = []
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        if child.name in EXCLUDED_DIRS:
            continue
        if child.name.startswith("."):
            continue
        video_count = sum(1 for p in child.iterdir() if is_video_file(p))
        if video_count <= 0:
            continue
        folders.append(child)
    return sorted(folders, key=lambda item: item.name.lower())


def _identity_map(memory: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    identities = memory.setdefault("identities", [])
    if not isinstance(identities, list):
        identities = []
        memory["identities"] = identities
    mapped: Dict[str, Dict[str, Any]] = {}
    for idx, item in enumerate(identities):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", "")).strip()
        if not label:
            continue
        normalized = _normalize_identity_record(item, label)
        identities[idx] = normalized
        mapped[label.lower()] = normalized
    return mapped


def _ensure_identity(memory: Dict[str, Any], label: str) -> Dict[str, Any]:
    mapped = _identity_map(memory)
    existing = mapped.get(label.lower())
    if existing is not None:
        if existing.get("label") != label:
            existing["label"] = label
        return existing

    identity = {
        "label": label,
        "prototype": [],
        "sample_count": 0,
        "confidence_sum": 0.0,
        "last_used": "",
        "locked": False,
        "locked_at": "",
        "positive_feedback_count": 0,
        "negative_feedback_count": 0,
        "correction_consistency_score": 0.0,
        "adaptive_auto_threshold": 0.82,
        "last_corrected_at": "",
        "reid_prototype": [],
        "reid_sample_count": 0,
        "reid_last_used": "",
        "reid_same_person_threshold": None,
    }
    identities = memory.setdefault("identities", [])
    if not isinstance(identities, list):
        identities = []
        memory["identities"] = identities
    identities.append(identity)
    return identity


def _find_identity(memory: Dict[str, Any], label: str) -> Optional[Dict[str, Any]]:
    return _identity_map(memory).get(label.lower())


def _safe_remove_empty_dir(path: Path) -> bool:
    try:
        path.rmdir()
        return True
    except Exception:
        return False


def move_video_collision_safe(src: Path, dst_dir: Path) -> Path:
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


def _merge_prototypes(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    target_proto = normalize_embedding(target.get("prototype", []))
    source_proto = normalize_embedding(source.get("prototype", []))
    target_count = int(target.get("sample_count", 0) or 0)
    source_count = int(source.get("sample_count", 0) or 0)

    if target_proto is not None and source_proto is not None and (target_count + source_count) > 0:
        merged = (target_proto * max(0, target_count)) + (source_proto * max(0, source_count))
        merged = normalize_embedding(merged.tolist())  # type: ignore[arg-type]
        if merged is not None:
            target["prototype"] = merged.tolist()
    elif source_proto is not None:
        target["prototype"] = source_proto.tolist()
    elif target_proto is not None:
        target["prototype"] = target_proto.tolist()

    target["sample_count"] = max(0, target_count) + max(0, source_count)
    target["confidence_sum"] = float(target.get("confidence_sum", 0.0) or 0.0) + float(
        source.get("confidence_sum", 0.0) or 0.0
    )
    target["positive_feedback_count"] = int(target.get("positive_feedback_count", 0) or 0) + int(
        source.get("positive_feedback_count", 0) or 0
    )
    target["negative_feedback_count"] = int(target.get("negative_feedback_count", 0) or 0) + int(
        source.get("negative_feedback_count", 0) or 0
    )
    target["locked"] = bool(target.get("locked", False)) or bool(source.get("locked", False))
    if bool(target.get("locked", False)) and not str(target.get("locked_at", "")).strip():
        target["locked_at"] = str(source.get("locked_at", "") or utc_now_iso())

    target_last = str(target.get("last_used", "") or "")
    source_last = str(source.get("last_used", "") or "")
    if source_last and (not target_last or source_last > target_last):
        target["last_used"] = source_last

    target_corrected = str(target.get("last_corrected_at", "") or "")
    source_corrected = str(source.get("last_corrected_at", "") or "")
    if source_corrected and (not target_corrected or source_corrected > target_corrected):
        target["last_corrected_at"] = source_corrected

    target_reid_proto = normalize_embedding(target.get("reid_prototype", []))
    source_reid_proto = normalize_embedding(source.get("reid_prototype", []))
    target_reid_count = int(target.get("reid_sample_count", 0) or 0)
    source_reid_count = int(source.get("reid_sample_count", 0) or 0)

    if target_reid_proto is not None and source_reid_proto is not None and (target_reid_count + source_reid_count) > 0:
        merged_reid = (target_reid_proto * max(0, target_reid_count)) + (source_reid_proto * max(0, source_reid_count))
        merged_reid = normalize_embedding(merged_reid.tolist())  # type: ignore[arg-type]
        if merged_reid is not None:
            target["reid_prototype"] = merged_reid.tolist()
    elif source_reid_proto is not None:
        target["reid_prototype"] = source_reid_proto.tolist()
    elif target_reid_proto is not None:
        target["reid_prototype"] = target_reid_proto.tolist()

    target["reid_sample_count"] = max(0, target_reid_count) + max(0, source_reid_count)
    target_reid_last = str(target.get("reid_last_used", "") or "")
    source_reid_last = str(source.get("reid_last_used", "") or "")
    if source_reid_last and (not target_reid_last or source_reid_last > target_reid_last):
        target["reid_last_used"] = source_reid_last
    if target.get("reid_same_person_threshold", None) is None and source.get("reid_same_person_threshold", None) is not None:
        target["reid_same_person_threshold"] = source.get("reid_same_person_threshold")


def _append_structural_event(
    memory: Dict[str, Any],
    *,
    action: str,
    from_label: str,
    to_label: str,
    source_path: str,
    final_path: str,
) -> None:
    record_structural_feedback(
        memory,
        action=action,
        source_action=action,
        from_label=from_label,
        to_label=to_label,
        source_path=source_path,
        final_path=final_path,
    )


def _replace_path_prefix(path_text: str, source_root: Path, target_root: Path) -> str:
    raw = str(path_text or "").strip()
    if not raw:
        return raw
    try:
        normalized_path = os.path.normpath(raw)
        source_text = os.path.normpath(str(source_root))
        if os.path.normcase(normalized_path) == os.path.normcase(source_text):
            return str(target_root)
        source_prefix = os.path.normcase(source_text + os.sep)
        candidate = os.path.normcase(normalized_path)
        if candidate.startswith(source_prefix):
            rel = normalized_path[len(source_text) :].lstrip("\\/")
            return str(target_root / rel)
    except Exception:
        return raw
    return raw


def list_identities(output_dir: Path, memory: Dict[str, Any]) -> Dict[str, Any]:
    mapped = _identity_map(memory)
    items: List[Dict[str, Any]] = []
    for folder in _iter_identity_folders(output_dir):
        video_count = sum(1 for p in folder.iterdir() if is_video_file(p))
        identity = mapped.get(folder.name.lower(), {})
        items.append(
            {
                "name": folder.name,
                "video_count": int(video_count),
                "locked": bool(identity.get("locked", False)),
                "memory_sample_count": int(identity.get("sample_count", 0) or 0),
                "last_used": str(identity.get("last_used", "") or ""),
            }
        )

    return {
        "ok": True,
        "identities": items,
        "count": len(items),
        "output_dir": str(output_dir),
    }


def _merge_identity_folders(
    output_dir: Path,
    memory: Dict[str, Any],
    source_folder: str,
    target_folder: str,
    embedding_cache_file: str = "",
) -> Dict[str, Any]:
    source_name = source_folder.strip()
    target_name = target_folder.strip()
    if not source_name or not target_name:
        raise RuntimeError("Both source and target folders are required for merge.")
    if source_name.lower() == target_name.lower():
        raise RuntimeError("Source and target folders must be different.")

    source_dir = output_dir / source_name
    target_dir = output_dir / target_name
    if not source_dir.is_dir():
        raise RuntimeError(f"Source identity folder does not exist: {source_dir}")
    if not target_dir.is_dir():
        raise RuntimeError(f"Target identity folder does not exist: {target_dir}")

    cache = _load_embedding_cache(output_dir, embedding_cache_file)
    cache_entries = cache.get("entries", {})
    if not isinstance(cache_entries, dict):
        cache_entries = {}
        cache["entries"] = cache_entries

    moved_map: Dict[str, str] = {}
    moved_count = 0
    for video in sorted(source_dir.iterdir(), key=lambda p: p.name.lower()):
        if not is_video_file(video):
            continue
        new_path = move_video_collision_safe(video, target_dir)
        moved_map[str(video.resolve())] = str(new_path.resolve())
        moved_count += 1

    removed_empty_source = False
    if source_dir.exists() and not any(source_dir.iterdir()):
        removed_empty_source = _safe_remove_empty_dir(source_dir)

    source_identity = _find_identity(memory, source_name)
    target_identity = _ensure_identity(memory, target_name)
    if source_identity is not None and source_identity is not target_identity:
        _merge_prototypes(target_identity, source_identity)
        identities = memory.setdefault("identities", [])
        if isinstance(identities, list):
            memory["identities"] = [row for row in identities if not (isinstance(row, dict) and str(row.get("label", "")).lower() == source_name.lower())]

    if bool(target_identity.get("locked", False)) and not str(target_identity.get("locked_at", "")).strip():
        target_identity["locked_at"] = utc_now_iso()

    source_root = source_dir.resolve()
    target_root = target_dir.resolve()
    moved_norm_map = {os.path.normcase(os.path.normpath(old)): new for old, new in moved_map.items()}

    decisions = memory.setdefault("decisions", [])
    if isinstance(decisions, list):
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            if str(decision.get("label", "")).strip().lower() == source_name.lower():
                decision["label"] = target_name
            if str(decision.get("memory_match_label", "")).strip().lower() == source_name.lower():
                decision["memory_match_label"] = target_name
            if str(decision.get("from_label", "")).strip().lower() == source_name.lower():
                decision["from_label"] = target_name
            if str(decision.get("to_label", "")).strip().lower() == source_name.lower():
                decision["to_label"] = target_name

            final_path = str(decision.get("final_path", "")).strip()
            if final_path:
                norm_final = os.path.normcase(os.path.normpath(final_path))
                if norm_final in moved_norm_map:
                    decision["final_path"] = moved_norm_map[norm_final]
                else:
                    decision["final_path"] = _replace_path_prefix(final_path, source_root, target_root)

    _append_structural_event(
        memory,
        action="identity_merge",
        from_label=source_name,
        to_label=target_name,
        source_path=str(source_dir),
        final_path=str(target_dir),
    )

    learning_feedback_with_embeddings = 0
    learning_feedback_events = 0
    now_iso = utc_now_iso()
    for source_path, final_path in moved_map.items():
        source_key = _normalize_cache_key(source_path)
        final_key = _normalize_cache_key(final_path)
        cache_entry_obj: Optional[Dict[str, Any]] = None
        raw_entry = cache_entries.pop(source_key, None)
        if isinstance(raw_entry, dict):
            cache_entry_obj = dict(raw_entry)
        elif isinstance(cache_entries.get(final_key), dict):
            cache_entry_obj = dict(cache_entries.get(final_key, {}))

        if cache_entry_obj is not None:
            cache_entry_obj["video_path"] = final_path
            cache_entry_obj["updated_at"] = now_iso
            cache_entries[final_key] = cache_entry_obj

        if _apply_learning_feedback_from_cache_entry(
            memory,
            source_action="identity_merge",
            from_label=source_name,
            to_label=target_name,
            source_path=source_path,
            final_path=final_path,
            cache_entry=cache_entry_obj,
        ):
            learning_feedback_with_embeddings += 1
            learning_feedback_events += 2 if source_name.lower() != target_name.lower() else 1

    _save_embedding_cache(output_dir, cache, embedding_cache_file)

    return {
        "source_folder": source_name,
        "target_folder": target_name,
        "moved_count": moved_count,
        "removed_empty_source": bool(removed_empty_source),
        "learning_feedback_with_embeddings": learning_feedback_with_embeddings,
        "learning_feedback_events": learning_feedback_events,
        "embedding_cache_file": str(_embedding_cache_path(output_dir, embedding_cache_file)),
    }


def _split_identity_folder(
    output_dir: Path,
    memory: Dict[str, Any],
    source_folder: str,
    target_folder: str,
    selected_videos: Sequence[str],
    embedding_cache_file: str = "",
) -> Dict[str, Any]:
    source_name = source_folder.strip()
    target_name = sanitize_folder_name(target_folder.strip())
    if not source_name:
        raise RuntimeError("Source folder is required for split.")
    if not target_name:
        raise RuntimeError("Target folder is required for split.")
    if source_name.lower() == target_name.lower():
        raise RuntimeError("Source and target folders must be different for split.")

    source_dir = output_dir / source_name
    target_dir = output_dir / target_name
    if not source_dir.is_dir():
        raise RuntimeError(f"Source identity folder does not exist: {source_dir}")

    cache = _load_embedding_cache(output_dir, embedding_cache_file)
    cache_entries = cache.get("entries", {})
    if not isinstance(cache_entries, dict):
        cache_entries = {}
        cache["entries"] = cache_entries

    candidate_paths: List[Path] = []
    for raw in selected_videos:
        path = Path(str(raw)).expanduser()
        if not path.exists():
            continue
        try:
            path.resolve().relative_to(source_dir.resolve())
        except Exception:
            raise RuntimeError(f"Selected video is not inside source folder: {path}")
        if not is_video_file(path):
            continue
        candidate_paths.append(path.resolve())

    if not candidate_paths:
        raise RuntimeError("No valid selected videos were provided for split.")

    moved_map: Dict[str, str] = {}
    for src in candidate_paths:
        new_path = move_video_collision_safe(src, target_dir)
        moved_map[str(src)] = str(new_path.resolve())

    _ensure_identity(memory, target_name)

    moved_norm_map = {os.path.normcase(os.path.normpath(old)): new for old, new in moved_map.items()}
    decisions = memory.setdefault("decisions", [])
    if isinstance(decisions, list):
        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            final_path = str(decision.get("final_path", "")).strip()
            if not final_path:
                continue
            norm_final = os.path.normcase(os.path.normpath(final_path))
            moved_to = moved_norm_map.get(norm_final)
            if moved_to:
                decision["final_path"] = moved_to
                decision["label"] = target_name
                decision["to_label"] = target_name

    _append_structural_event(
        memory,
        action="identity_split",
        from_label=source_name,
        to_label=target_name,
        source_path=str(source_dir),
        final_path=str(target_dir),
    )

    learning_feedback_with_embeddings = 0
    learning_feedback_events = 0
    now_iso = utc_now_iso()
    for source_path, final_path in moved_map.items():
        source_key = _normalize_cache_key(source_path)
        final_key = _normalize_cache_key(final_path)
        cache_entry_obj: Optional[Dict[str, Any]] = None
        raw_entry = cache_entries.pop(source_key, None)
        if isinstance(raw_entry, dict):
            cache_entry_obj = dict(raw_entry)
        elif isinstance(cache_entries.get(final_key), dict):
            cache_entry_obj = dict(cache_entries.get(final_key, {}))

        if cache_entry_obj is not None:
            cache_entry_obj["video_path"] = final_path
            cache_entry_obj["updated_at"] = now_iso
            cache_entries[final_key] = cache_entry_obj

        if _apply_learning_feedback_from_cache_entry(
            memory,
            source_action="identity_split",
            from_label=source_name,
            to_label=target_name,
            source_path=source_path,
            final_path=final_path,
            cache_entry=cache_entry_obj,
        ):
            learning_feedback_with_embeddings += 1
            learning_feedback_events += 2 if source_name.lower() != target_name.lower() else 1

    _save_embedding_cache(output_dir, cache, embedding_cache_file)

    return {
        "source_folder": source_name,
        "target_folder": target_name,
        "moved_count": len(moved_map),
        "moved_videos": list(moved_map.values()),
        "learning_feedback_with_embeddings": learning_feedback_with_embeddings,
        "learning_feedback_events": learning_feedback_events,
        "embedding_cache_file": str(_embedding_cache_path(output_dir, embedding_cache_file)),
    }


def _lock_identity(output_dir: Path, memory: Dict[str, Any], folder_name: str, lock_value: bool) -> Dict[str, Any]:
    label = folder_name.strip()
    if not label:
        raise RuntimeError("Identity folder name is required.")
    folder = output_dir / label
    if not folder.is_dir():
        raise RuntimeError(f"Identity folder does not exist: {folder}")

    identity = _ensure_identity(memory, label)
    identity["locked"] = bool(lock_value)
    identity["locked_at"] = utc_now_iso() if lock_value else ""
    identity["last_used"] = str(identity.get("last_used", "") or "")

    _append_structural_event(
        memory,
        action="identity_lock" if lock_value else "identity_unlock",
        from_label=label,
        to_label=label,
        source_path=str(folder),
        final_path=str(folder),
    )

    return {
        "folder": label,
        "locked": bool(lock_value),
    }


def perform_identity_action(
    *,
    output_dir: Path,
    memory_path: Path,
    embedding_cache_file: str = "",
    action: str,
    source_folder: str = "",
    target_folder: str = "",
    folder_name: str = "",
    selected_videos: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    action_key = action.strip().lower()
    memory = load_memory(memory_path)

    if action_key == "merge":
        details = _merge_identity_folders(
            output_dir,
            memory,
            source_folder,
            target_folder,
            embedding_cache_file=embedding_cache_file,
        )
    elif action_key == "split":
        details = _split_identity_folder(
            output_dir,
            memory,
            source_folder,
            target_folder,
            selected_videos or [],
            embedding_cache_file=embedding_cache_file,
        )
    elif action_key == "lock":
        details = _lock_identity(output_dir, memory, folder_name, True)
    elif action_key == "unlock":
        details = _lock_identity(output_dir, memory, folder_name, False)
    else:
        raise RuntimeError(f"Unsupported identity action: {action}")

    refresh_all_identity_stats(memory)
    save_memory(memory_path, memory)
    return {
        "ok": True,
        "action": action_key,
        "details": details,
        "memory_file": str(memory_path),
    }
