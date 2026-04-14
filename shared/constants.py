"""Single source of truth for constants shared across all modules."""

from __future__ import annotations

from typing import Dict, Set, Tuple

# ---------------------------------------------------------------------------
# Video file extensions recognised throughout the application.
# ---------------------------------------------------------------------------
VIDEO_EXTS: Set[str] = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

# ---------------------------------------------------------------------------
# Directories that the scanner should always ignore when doing recursive scans.
# ---------------------------------------------------------------------------
EXCLUDED_SCAN_DIRS: Set[str] = {
    ".model_cache",
    ".learning",
    ".reports",
    "__pycache__",
}

# ---------------------------------------------------------------------------
# Well-known output directory names produced by the sorter.
# ---------------------------------------------------------------------------
UNCERTAIN_DIRNAME: str = "Uncertain"
REPORTS_DIRNAME: str = ".reports"
LEARNING_DIRNAME: str = ".learning"
DUPLICATES_DIRNAME: str = "Duplicates"
NO_FEMALE_DIRNAME: str = "No_Female_Found"

# ---------------------------------------------------------------------------
# Embedding cache constants.
# ---------------------------------------------------------------------------
SORTING_EMBEDDING_CACHE_FILENAME: str = "video_embedding_cache.json"
SORTING_EMBEDDING_CACHE_INSIGHTFACE_FILENAME: str = "video_embedding_cache_insightface.json"
SORTING_EMBEDDING_CACHE_FACENET_FILENAME: str = "video_embedding_cache_facenet.json"
SORTING_EMBEDDING_CACHE_SCHEMA_VERSION: int = 1
SORTING_EMBEDDING_CACHE_MAX_ITEMS: int = 50000

# ---------------------------------------------------------------------------
# Learning memory filenames per engine.
# ---------------------------------------------------------------------------
LEARNING_MEMORY_INSIGHTFACE_FILENAME: str = "memory_insightface_v1.json"
LEARNING_MEMORY_FACENET_FILENAME: str = "memory_facenet_v1.json"

# ---------------------------------------------------------------------------
# Gender classifier labels and model assets.
# ---------------------------------------------------------------------------
GENDER_LABELS = ["Male", "Female"]
GENDER_MEAN: Tuple[float, float, float] = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_PROTO_URL: str = (
    "https://raw.githubusercontent.com/spmallick/learnopencv/master/AgeGender/gender_deploy.prototxt"
)
GENDER_MODEL_URLS = [
    "https://raw.githubusercontent.com/smahesh29/Gender-and-Age-Detection/master/gender_net.caffemodel",
    "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_net.caffemodel",
]

# ---------------------------------------------------------------------------
# Profile names.
# ---------------------------------------------------------------------------
PROFILE_FAST: str = "fast"
PROFILE_BALANCED: str = "balanced"
PROFILE_HIGH_ACCURACY: str = "high_accuracy"

# Flag-to-CLI-token mapping used by profile application logic.
PROFILE_KEY_TO_FLAG: Dict[str, str] = {
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

# ---------------------------------------------------------------------------
# Result / communication constants.
# ---------------------------------------------------------------------------
RESULT_JSON_PREFIX: str = "[RESULT_JSON] "
PREVIEW_WINDOW_TITLE: str = "Live Frame Preview"

# ---------------------------------------------------------------------------
# Explainability tags in priority order.
# ---------------------------------------------------------------------------
REASON_TAG_PRIORITY = [
    "memory_match_applied",
    "gender_disagreement",
    "few_stable_embeddings",
    "few_votes",
    "low_face_area",
    "unverified_female_candidates",
    "memory_match_suggested",
]

# Retry checkpoint ratios (5 %, 10 %, …, 95 %).
RETRY_CHECKPOINT_RATIOS: Tuple[float, ...] = tuple(
    step / 100.0 for step in range(5, 100, 5)
)
