import tempfile
import unittest
from pathlib import Path

from sort_videos_by_female_faces_gpu import UNCERTAIN_DIRNAME, list_videos, move_video


class UncertainReprocessTests(unittest.TestCase):
    def test_list_videos_excludes_uncertain_bucket(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "root_clip.mp4").write_bytes(b"a")
            (root / UNCERTAIN_DIRNAME).mkdir(parents=True, exist_ok=True)
            (root / UNCERTAIN_DIRNAME / "u.mp4").write_bytes(b"b")
            (root / "No_Female_Found").mkdir(parents=True, exist_ok=True)
            (root / "No_Female_Found" / "n.mp4").write_bytes(b"c")
            (root / "Female_1").mkdir(parents=True, exist_ok=True)
            (root / "Female_1" / "f.mp4").write_bytes(b"d")

            normal = [p.name for p in list_videos(str(root), recursive=True, include_generated_folders=False)]
            self.assertEqual(normal, ["root_clip.mp4"])

            with_generated = [p.name for p in list_videos(str(root), recursive=True, include_generated_folders=True)]
            self.assertIn("root_clip.mp4", with_generated)
            self.assertIn("f.mp4", with_generated)
            self.assertIn("n.mp4", with_generated)
            self.assertNotIn("u.mp4", with_generated)

    def test_move_video_is_noop_for_same_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bucket = Path(tmp) / UNCERTAIN_DIRNAME
            bucket.mkdir(parents=True, exist_ok=True)
            src = bucket / "clip.mp4"
            src.write_bytes(b"video")

            moved = move_video(src, bucket)
            self.assertEqual(moved, src)
            self.assertTrue(src.exists())
            self.assertFalse((bucket / "clip_1.mp4").exists())


if __name__ == "__main__":
    unittest.main()

