import shutil
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from duplicate_tools import _choose_primary, apply_duplicate_move, scan_duplicates
from learning_memory import default_memory_path, load_memory


def _write_test_video(path: Path, size: tuple[int, int], variant: str) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 8.0, size)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer for {path}")
    try:
        width, height = size
        for idx in range(16):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            if variant == "base":
                frame[:] = (30, 90, 180)
                cv2.rectangle(frame, (idx * 2 % width, 8), (min(width - 1, idx * 2 % width + 18), 24), (255, 255, 255), -1)
            elif variant == "near":
                frame[:] = (34, 95, 184)
                cv2.rectangle(frame, (idx * 2 % width, 8), (min(width - 1, idx * 2 % width + 18), 24), (245, 245, 245), -1)
            else:
                noise = np.random.default_rng(seed=idx).integers(0, 255, size=(height, width, 3), dtype=np.uint8)
                frame = noise
            writer.write(frame)
    finally:
        writer.release()


class DuplicateToolsTests(unittest.TestCase):
    def test_exact_duplicate_grouping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            folder = output_dir / "Female_A"
            folder.mkdir(parents=True, exist_ok=True)
            src = folder / "clip_a.avi"
            dup = folder / "clip_b.avi"
            other = folder / "clip_c.avi"
            _write_test_video(src, (64, 48), "base")
            shutil.copyfile(src, dup)
            _write_test_video(other, (64, 48), "other")

            payload = scan_duplicates(output_dir)
            self.assertTrue(payload["ok"])
            groups = payload.get("groups", [])
            exact_groups = [g for g in groups if g.get("match_type") == "exact"]
            self.assertEqual(len(exact_groups), 1)
            group = exact_groups[0]
            self.assertEqual(int(group.get("duplicate_count", 0)), 1)

    def test_near_duplicate_grouping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            folder = output_dir / "Female_B"
            folder.mkdir(parents=True, exist_ok=True)
            a = folder / "a.avi"
            b = folder / "b.avi"
            c = folder / "c.avi"
            _write_test_video(a, (72, 54), "base")
            _write_test_video(b, (72, 54), "near")
            _write_test_video(c, (72, 54), "other")

            payload = scan_duplicates(output_dir)
            groups = payload.get("groups", [])
            near_groups = [g for g in groups if g.get("match_type") == "near"]
            self.assertGreaterEqual(len(near_groups), 1)
            near_paths = set()
            for g in near_groups:
                near_paths.add(str(g.get("primary_path", "")))
                for path in g.get("duplicate_paths", []):
                    near_paths.add(str(path))
            self.assertIn(str(a.resolve()), near_paths)
            self.assertIn(str(b.resolve()), near_paths)

    def test_choose_primary_prefers_higher_quality(self) -> None:
        items = [
            {
                "path": "C:/a.mp4",
                "width": 1280,
                "height": 720,
                "estimated_bitrate": 1200.0,
                "size": 2_000_000,
            },
            {
                "path": "C:/b.mp4",
                "width": 1920,
                "height": 1080,
                "estimated_bitrate": 1100.0,
                "size": 2_100_000,
            },
        ]
        primary = _choose_primary(items)
        self.assertEqual(primary["path"], "C:/b.mp4")

    def test_apply_duplicate_move_collision_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            src_dir = output_dir / "Female_C"
            src_dir.mkdir(parents=True, exist_ok=True)
            src_file = src_dir / "clip.avi"
            _write_test_video(src_file, (64, 48), "base")

            dup_target = output_dir / "Duplicates" / "Female_C"
            dup_target.mkdir(parents=True, exist_ok=True)
            existing = dup_target / "clip.avi"
            _write_test_video(existing, (64, 48), "near")

            payload = apply_duplicate_move(output_dir, {"paths": [str(src_file)]})
            self.assertTrue(payload["ok"])
            details = payload["details"]
            self.assertEqual(int(details["moved_count"]), 1)
            moved_to = Path(details["moved"][0]["destination_path"])
            self.assertTrue(moved_to.exists())
            self.assertNotEqual(moved_to.name, "clip.avi")

    def test_apply_duplicate_move_emits_structural_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            src_dir = output_dir / "Female_E"
            src_dir.mkdir(parents=True, exist_ok=True)
            src_file = src_dir / "dup.avi"
            _write_test_video(src_file, (64, 48), "base")
            memory_path = default_memory_path(output_dir)

            payload = apply_duplicate_move(
                output_dir,
                {"paths": [str(src_file)]},
                memory_path=memory_path,
            )
            self.assertTrue(payload["ok"])
            memory = load_memory(memory_path)
            self.assertTrue(
                any(
                    str(item.get("source_action", "")) == "duplicates_move"
                    and str(item.get("feedback_event_type", "")) == "structural"
                    for item in memory.get("decisions", [])
                )
            )


if __name__ == "__main__":
    unittest.main()
