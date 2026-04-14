# Changelog

All notable changes to the AI Video Sorter project will be documented in this file.

## [Unreleased] – 2026-04-14

### Architecture (Phase 1 + Phase 2)

#### Added
- **`shared/` package** – Single source of truth for constants, utilities, and embedding operations
  - `shared/constants.py` – All constants previously duplicated across 4-5 modules
  - `shared/utils.py` – Canonical `safe_int`, `safe_float`, `ensure_dir`, `move_video_collision_safe`, `utc_now_iso`, `atomic_write_text`, etc.
  - `shared/embedding_ops.py` – Canonical `cosine_similarity`, `normalize_embedding`, `robust_average_embeddings`
  - `shared/py.typed` – PEP 561 marker for type checking
- **`reporting.py`** – Run summary payloads, per-video rows, CSV/JSON report generation (extracted from main module)
- **`embedding_cache.py`** – Sorted embedding cache load/save/persist logic (extracted from main module)
- **`clustering.py`** – DBSCAN-based identity clustering with merge logic (extracted from main module)
- **`explainability.py`** – Result records, reason tags, JSON payloads, explainability metadata (extracted from main module)
- **`memory_assist.py`** – Memory-assisted identity matching and review learning (extracted from main module)
- **`test_integration.py`** – Integration test with synthetic video fixtures (new)

#### Changed
- `sort_videos_by_female_faces_gpu.py` – Reduced from 3,442 to ~3,050 lines; imports extracted modules and re-exports symbols for backward compatibility
- `requirements.txt` – Added missing `Pillow>=10.0.0` dependency

#### Migration Notes
- All imports from `sort_videos_by_female_faces_gpu` continue to work (re-exported)
- No CLI or GUI changes required
- No configuration changes required

### Performance (Phase 3)

#### Optimized
- **`build_sample_times()`** – Replaced float-accumulating `while` loop with `np.arange` to prevent drift
- **`stabilize_identity()`** – Incremental running mean (O(1) per step) replaces full `np.vstack` + re-average (O(n) per step)
- **`_phash_frame()`** – Vectorized bit-packing via `np.packbits` (~10× faster than Python loop over 64 bits)
- **`_build_near_groups()`** – Duration pre-bucketing reduces near-duplicate comparison from O(n²) to O(n × k) where k ≪ n

### Hardening (Phase 4)

#### Added
- `py.typed` marker for shared package (PEP 561)
- Integration test with synthetic video fixtures
- `CHANGELOG.md` with migration notes
- Docstrings for all public functions in extracted modules
