from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from folder_naming import sanitize_folder_name


REVIEW_STATE_SCHEMA_VERSION = 1
REVIEW_STATE_DIRNAME = ".review_queue"
REVIEW_STATE_FILENAME = "review_state.json"
REVIEW_PENDING_DIRNAME = "Review_Pending"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def review_state_path(output_dir: Path) -> Path:
    return output_dir / REVIEW_STATE_DIRNAME / REVIEW_STATE_FILENAME


def save_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def load_review_state(output_dir: Path) -> Dict[str, Any]:
    path = review_state_path(output_dir)
    if not path.exists():
        return {
            "schema_version": REVIEW_STATE_SCHEMA_VERSION,
            "run_id": "",
            "created_at": "",
            "updated_at": "",
            "items": [],
        }
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "schema_version": REVIEW_STATE_SCHEMA_VERSION,
            "run_id": "",
            "created_at": "",
            "updated_at": "",
            "items": [],
        }

    if not isinstance(data, dict):
        return {
            "schema_version": REVIEW_STATE_SCHEMA_VERSION,
            "run_id": "",
            "created_at": "",
            "updated_at": "",
            "items": [],
        }

    items = data.get("items")
    if not isinstance(items, list):
        data["items"] = []
    data.setdefault("schema_version", REVIEW_STATE_SCHEMA_VERSION)
    data.setdefault("run_id", "")
    data.setdefault("created_at", "")
    data.setdefault("updated_at", "")
    return data


def save_review_state(output_dir: Path, state: Dict[str, Any]) -> None:
    state["schema_version"] = REVIEW_STATE_SCHEMA_VERSION
    state["updated_at"] = utc_now_iso()
    save_json_atomic(review_state_path(output_dir), state)


def create_review_state(
    output_dir: Path,
    queue_items: List[Dict[str, Any]],
    run_id: str,
) -> Dict[str, Any]:
    now = utc_now_iso()
    return {
        "schema_version": REVIEW_STATE_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": now,
        "updated_at": now,
        "items": queue_items,
        "output_dir": str(output_dir),
    }


def pending_review_items(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = state.get("items", [])
    if not isinstance(items, list):
        return []
    pending: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or item.get("status") != "pending":
            continue
        pending_path = Path(str(item.get("pending_path", "")).strip())
        if pending_path.exists():
            pending.append(item)
    return pending


def move_video_collision_safe(src: Path, dst_dir: Path) -> Path:
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


def decide_review_route(
    result: Dict[str, Any],
    review_mode: bool,
    review_confidence_threshold: float = 0.75,
) -> str:
    label = str(result.get("decision_label", "")).strip().lower()
    if not review_mode:
        return "direct"
    if label == "uncertain":
        return "queue"
    if label == "female_detected":
        try:
            confidence = float(result.get("confidence_score", 0.0))
        except Exception:
            confidence = 0.0
        if confidence < float(review_confidence_threshold):
            return "queue"
        return "direct"
    if label == "no_female":
        return "direct_no_female"
    return "direct"


def apply_review_action(
    output_dir: Path,
    item_id: str,
    action: str,
    target_folder: str = "",
) -> Dict[str, Any]:
    state = load_review_state(output_dir)
    items = state.get("items", [])
    if not isinstance(items, list):
        raise RuntimeError("Invalid review state: items is not a list.")

    item: Optional[Dict[str, Any]] = None
    for candidate in items:
        if isinstance(candidate, dict) and str(candidate.get("id", "")) == item_id:
            item = candidate
            break

    if item is None:
        raise RuntimeError(f"Review item not found: {item_id}")
    if item.get("status") != "pending":
        raise RuntimeError(f"Review item is not pending: {item_id}")

    pending_path = Path(str(item.get("pending_path", "")))
    if action != "skip" and not pending_path.exists():
        raise RuntimeError(f"Pending video is missing: {pending_path}")

    action_key = action.strip().lower()
    final_path: Path
    review_status: str

    if action_key == "approve_suggested":
        suggested = str(item.get("suggested_folder", "")).strip()
        folder = sanitize_folder_name(suggested) if suggested else "Female_Unknown"
        final_path = move_video_collision_safe(pending_path, output_dir / folder)
        review_status = "approved"
    elif action_key == "move_no_female":
        final_path = move_video_collision_safe(pending_path, output_dir / "No_Female_Found")
        review_status = "moved_no_female"
    elif action_key == "reassign_existing":
        if not target_folder.strip():
            raise RuntimeError("Existing folder name is required.")
        folder = sanitize_folder_name(target_folder.strip())
        existing_dir = output_dir / folder
        if not existing_dir.is_dir():
            raise RuntimeError(f"Target folder does not exist: {existing_dir}")
        final_path = move_video_collision_safe(pending_path, existing_dir)
        review_status = "reassigned"
    elif action_key == "reassign_new":
        if not target_folder.strip():
            raise RuntimeError("New folder name is required.")
        folder = sanitize_folder_name(target_folder.strip())
        final_path = move_video_collision_safe(pending_path, output_dir / folder)
        review_status = "reassigned"
    elif action_key == "skip":
        final_path = pending_path
        review_status = "skipped"
    else:
        raise RuntimeError(f"Unsupported review action: {action}")

    item["status"] = review_status
    item["final_path"] = str(final_path)
    item["final_label"] = final_path.parent.name
    item["reviewed_at"] = utc_now_iso()
    item["review_action"] = action_key
    if action_key in {"reassign_existing", "reassign_new"}:
        item["review_target_folder"] = sanitize_folder_name(target_folder.strip())

    save_review_state(output_dir, state)
    return item
