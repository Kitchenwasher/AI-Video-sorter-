"""Integration tests using synthetic video fixtures.

These tests verify the end-to-end pipeline works correctly from video listing
through explainability metadata, reporting, and embedding cache persistence
without requiring real face models or GPU hardware.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from shared.constants import UNCERTAIN_DIRNAME, REPORTS_DIRNAME, LEARNING_DIRNAME
from shared.utils import ensure_dir, utc_now_iso, move_video_collision_safe, clamp_confidence
from shared.embedding_ops import cosine_similarity, normalize_embedding, robust_average_embeddings

from clustering import cluster_embeddings
from embedding_cache import (
    load_sorting_embedding_cache,
    persist_sorted_embedding_cache,
    sorting_embedding_cache_filename,
)
from explainability import (
    apply_explainability_metadata,
    build_result_json_payload,
    new_result_record,
    set_decision,
    emit_progress,
)
from reporting import (
    build_per_video_report_rows,
    build_run_summary_payload,
    report_output_dir,
    write_run_reports,
)

# ---------------------------------------------------------------------------
# Synthetic video fixture helpers
# ---------------------------------------------------------------------------

def _create_synthetic_video(path: Path, *, width: int = 320, height: int = 240,
                            fps: float = 25.0, duration_sec: float = 2.0,
                            color: tuple = (0, 128, 255)) -> Path:
    """Create a tiny synthetic video with solid-colored frames."""
    ensure_dir(path.parent)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    total_frames = int(fps * duration_sec)
    frame = np.full((height, width, 3), color, dtype=np.uint8)
    for _ in range(total_frames):
        writer.write(frame)
    writer.release()
    return path


def _make_fake_embedding(dim: int = 512, seed: int = 0) -> List[float]:
    """Create a deterministic fake embedding vector."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-8
    return vec.tolist()


def _make_result(video_path: str, label: str, confidence: float,
                 seed: int = 0, folder_name: str = "") -> Dict[str, Any]:
    """Build a complete mock result record for pipeline testing."""
    result = new_result_record(video_path, "CPU fallback")
    set_decision(result, label, f"Synthetic: {label}", confidence)
    result["embedding"] = _make_fake_embedding(seed=seed)
    result["suggested_folder_name"] = folder_name or label
    result["reason_metrics"] = {
        "total_faces_evaluated": 10,
        "low_face_area_rejections": 2,
        "stable_embeddings": 5,
        "gender_votes": 4,
        "female_score": 2.1,
        "male_score": 0.3,
        "female_seed_hits": 3,
        "best_seed_confidence": 0.89,
    }
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class SyntheticVideoCreationTest(unittest.TestCase):
    """Verify we can create and read back synthetic video fixtures."""

    def test_synthetic_video_is_readable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video_path = _create_synthetic_video(
                Path(tmp) / "test_clip.mp4", duration_sec=1.0
            )
            self.assertTrue(video_path.exists())
            cap = cv2.VideoCapture(str(video_path))
            try:
                self.assertTrue(cap.isOpened())
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.assertGreater(total, 0)
                ok, frame = cap.read()
                self.assertTrue(ok)
                self.assertIsNotNone(frame)
                self.assertEqual(frame.shape, (240, 320, 3))
            finally:
                cap.release()


class EndToEndPipelineTest(unittest.TestCase):
    """Verify the full pipeline from results through reporting and caching."""

    def test_full_pipeline_flow(self) -> None:
        """Integration: results → explainability → reporting → embedding cache."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)

            # 1. Create synthetic videos and mock results
            videos = []
            results = []
            for i in range(5):
                vpath = _create_synthetic_video(
                    output_dir / f"Female_{i + 1}" / f"clip_{i}.mp4",
                    color=(i * 50, 100, 200),
                    duration_sec=1.0,
                )
                videos.append(vpath)
                result = _make_result(
                    str(vpath),
                    "female_detected",
                    0.85 + i * 0.02,
                    seed=i,
                    folder_name=f"Female_{i + 1}",
                )
                result["final_destination"] = str(vpath)
                results.append(result)

            # Add an uncertain result
            uncertain_vid = _create_synthetic_video(
                output_dir / UNCERTAIN_DIRNAME / "uncertain.mp4",
                color=(200, 50, 50),
            )
            uncertain_result = _make_result(
                str(uncertain_vid), "uncertain", 0.55, seed=99
            )
            uncertain_result["reason_metrics"]["stable_embeddings"] = 1
            uncertain_result["reason_metrics"]["gender_votes"] = 1
            results.append(uncertain_result)

            # 2. Apply explainability metadata to all results
            class FakeCfg:
                min_stable_embeddings = 3
                min_stabilization_gender_votes = 3

            cfg = FakeCfg()
            for r in results:
                apply_explainability_metadata(r, cfg)

            # Verify explainability was applied
            for r in results:
                self.assertIn("reason_tags", r)
                self.assertIn("reason_summary", r)
                self.assertIsInstance(r["reason_tags"], list)
                self.assertTrue(len(r["reason_summary"]) > 0)

            # The uncertain result should have tags
            self.assertGreater(len(results[-1]["reason_tags"]), 0)

            # 3. Build JSON payloads
            payloads = [build_result_json_payload(r) for r in results]
            for p in payloads:
                self.assertIn("video", p)
                self.assertIn("decision_label", p)
                self.assertIn("confidence_score", p)
                self.assertIn("reason_tags", p)

            # 4. Generate reports
            from datetime import datetime, timezone, timedelta
            run_id = "20260414T120000Z-test1234"
            start = datetime(2026, 4, 14, 12, 0, 0, tzinfo=timezone.utc)
            finish = start + timedelta(seconds=5.5)

            summary = build_run_summary_payload(
                run_id=run_id,
                run_started_at=start,
                run_finished_at=finish,
                total_scanned=6,
                processed_successfully=5,
                female_detected=5,
                uncertain=1,
                no_female_found=0,
                errors=0,
                stopped_early=False,
            )
            video_rows = build_per_video_report_rows(results)
            report_paths = write_run_reports(
                output_dir=output_dir,
                run_id=run_id,
                summary_payload=summary,
                video_rows=video_rows,
            )

            # Verify reports were written
            for key in ("summary_json", "summary_csv", "videos_json", "videos_csv"):
                self.assertTrue(Path(report_paths[key]).exists(), f"Missing: {key}")

            # Verify report content
            summary_data = json.loads(
                Path(report_paths["summary_json"]).read_text(encoding="utf-8")
            )
            self.assertEqual(summary_data["total_scanned"], 6)
            self.assertEqual(summary_data["female_detected"], 5)

            # 5. Persist embedding cache
            cache_info = persist_sorted_embedding_cache(
                output_dir, results, use_insightface=True
            )
            self.assertGreater(cache_info["updated_entries"], 0)

            # Verify cache can be loaded back
            cache = load_sorting_embedding_cache(
                output_dir / LEARNING_DIRNAME / sorting_embedding_cache_filename(True)
            )
            self.assertGreater(len(cache.get("entries", {})), 0)


class ClusteringIntegrationTest(unittest.TestCase):
    """Verify embedding clustering produces sane results."""

    def test_cluster_similar_embeddings_together(self) -> None:
        """Close embeddings should cluster together; distant ones apart."""
        # Two "identities" — seeds 0-2 are similar, seeds 100-102 are similar
        rng1 = np.random.RandomState(42)
        base_a = rng1.randn(512).astype(np.float32)
        base_a /= np.linalg.norm(base_a)

        rng2 = np.random.RandomState(99)
        base_b = rng2.randn(512).astype(np.float32)
        base_b /= np.linalg.norm(base_b)

        results = []
        for i in range(3):
            noise = rng1.randn(512).astype(np.float32) * 0.02
            emb = base_a + noise
            emb /= np.linalg.norm(emb)
            results.append({"video": f"group_a_{i}.mp4", "embedding": emb.tolist()})

        for i in range(3):
            noise = rng2.randn(512).astype(np.float32) * 0.02
            emb = base_b + noise
            emb /= np.linalg.norm(emb)
            results.append({"video": f"group_b_{i}.mp4", "embedding": emb.tolist()})

        assignments = cluster_embeddings(
            results, eps=0.3, min_samples=1, cluster_merge_threshold=0.85
        )

        # All group_a videos should have the same cluster ID
        a_ids = {assignments[f"group_a_{i}.mp4"] for i in range(3)}
        b_ids = {assignments[f"group_b_{i}.mp4"] for i in range(3)}
        self.assertEqual(len(a_ids), 1, "Group A should be one cluster")
        self.assertEqual(len(b_ids), 1, "Group B should be one cluster")
        self.assertNotEqual(a_ids, b_ids, "Groups A and B should be different clusters")


class EmbeddingOpsIntegrationTest(unittest.TestCase):
    """Verify embedding operations work correctly with real-ish data."""

    def test_normalize_embedding_idempotent(self) -> None:
        raw = _make_fake_embedding(seed=7)
        normalized = normalize_embedding(raw)
        self.assertIsNotNone(normalized)
        norm = float(np.linalg.norm(normalized))
        self.assertAlmostEqual(norm, 1.0, places=5)

        # Normalizing again should return same result
        renormalized = normalize_embedding(normalized.tolist())
        self.assertIsNotNone(renormalized)
        self.assertAlmostEqual(
            cosine_similarity(normalized, renormalized), 1.0, places=5
        )

    def test_robust_average_excludes_outliers(self) -> None:
        rng = np.random.RandomState(10)
        base = rng.randn(512).astype(np.float32)
        base /= np.linalg.norm(base)

        embeddings = []
        for _ in range(9):
            noise = rng.randn(512).astype(np.float32) * 0.01
            emb = base + noise
            emb /= np.linalg.norm(emb)
            embeddings.append(emb)

        # Add one outlier
        outlier = rng.randn(512).astype(np.float32)
        outlier /= np.linalg.norm(outlier)
        embeddings.append(outlier)

        avg = robust_average_embeddings(embeddings)
        # Average should be close to base, not affected much by outlier
        sim = cosine_similarity(avg, base)
        self.assertGreater(sim, 0.95)

    def test_cosine_similarity_range(self) -> None:
        a = np.array(_make_fake_embedding(seed=1))
        b = np.array(_make_fake_embedding(seed=2))
        sim = cosine_similarity(a, b)
        self.assertGreaterEqual(sim, -1.0)
        self.assertLessEqual(sim, 1.0)

        # Self-similarity should be ~1.0
        self_sim = cosine_similarity(a, a)
        self.assertAlmostEqual(self_sim, 1.0, places=5)


class MoveVideoCollisionTest(unittest.TestCase):
    """Verify collision-safe file moving works correctly."""

    def test_collision_generates_unique_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            src_dir = Path(tmp) / "source"
            dst_dir = Path(tmp) / "dest"
            ensure_dir(src_dir)
            ensure_dir(dst_dir)

            # Create "existing" file at destination
            (dst_dir / "clip.mp4").write_bytes(b"existing")

            # Create source file
            src = src_dir / "clip.mp4"
            src.write_bytes(b"new_video")

            moved = move_video_collision_safe(src, dst_dir)
            self.assertNotEqual(moved.name, "clip.mp4")
            self.assertTrue(moved.name.startswith("clip_"))
            self.assertTrue(moved.exists())
            self.assertEqual(moved.read_bytes(), b"new_video")


class UtilityFunctionsTest(unittest.TestCase):
    """Verify shared utility function correctness."""

    def test_clamp_confidence_bounds(self) -> None:
        self.assertEqual(clamp_confidence(-0.5), 0.0)
        self.assertEqual(clamp_confidence(1.5), 1.0)
        self.assertEqual(clamp_confidence(0.75), 0.75)

    def test_utc_now_iso_format(self) -> None:
        ts = utc_now_iso()
        self.assertIn("T", ts)
        self.assertTrue(ts.endswith("+00:00"))


if __name__ == "__main__":
    unittest.main()
