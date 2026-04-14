from __future__ import annotations

import collections
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence


NAME_STOPWORDS = {
    "video",
    "vid",
    "clip",
    "movie",
    "scene",
    "part",
    "take",
    "edit",
    "final",
    "draft",
    "new",
    "copy",
    "female",
    "girl",
    "girls",
    "woman",
    "women",
    "face",
    "faces",
    "sorted",
}


def sanitize_folder_name(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\\\|?*]+', "_", name).strip(" .")
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned or "Unknown"


def infer_name_from_video_paths(video_paths: Sequence[str]) -> Optional[str]:
    stems = [Path(path).stem.lower() for path in video_paths]
    if not stems:
        return None

    token_counter: collections.Counter[str] = collections.Counter()
    for stem in stems:
        tokens = re.split(r"[^a-z0-9]+", stem)
        valid = {
            token
            for token in tokens
            if len(token) >= 3 and not token.isdigit() and token not in NAME_STOPWORDS
        }
        token_counter.update(valid)

    if not token_counter:
        return None

    min_occurrences = max(2, (len(stems) + 1) // 2)
    ranked = sorted(token_counter.items(), key=lambda item: (item[1], len(item[0]), item[0]), reverse=True)
    for token, count in ranked:
        if count >= min_occurrences:
            return token.title()
    return None


def build_cluster_folder_names(cluster_map: Dict[str, int]) -> Dict[int, str]:
    grouped: Dict[int, List[str]] = {}
    for video_path, cluster_id in cluster_map.items():
        grouped.setdefault(int(cluster_id), []).append(video_path)

    used_names: Dict[str, int] = {}
    folder_names: Dict[int, str] = {}
    for cluster_id in sorted(grouped.keys()):
        inferred = infer_name_from_video_paths(grouped[cluster_id])
        base_name = f"Female_{inferred}" if inferred else f"Female_{cluster_id}"
        base_name = sanitize_folder_name(base_name)
        if base_name.lower() == "no_female_found":
            base_name = f"Female_{cluster_id}"

        final_name = base_name
        suffix = 2
        while final_name.lower() in used_names:
            final_name = f"{base_name}_{suffix}"
            suffix += 1

        used_names[final_name.lower()] = cluster_id
        folder_names[cluster_id] = final_name

    return folder_names
