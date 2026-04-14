import tempfile
import unittest
from pathlib import Path

from sort_videos_by_female_faces_gpu import (
    UNCERTAIN_DIRNAME,
    Config,
    build_uncertain_retry_checkpoints,
    list_videos,
    move_video,
    retry_find_first_female,
)


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

    def test_retry_checkpoints_follow_5_percent_steps(self) -> None:
        checkpoints = build_uncertain_retry_checkpoints(
            seed_hits=2,
            duration_sec=100.0,
            initial_scan_end_sec=20.0,
        )
        self.assertEqual(checkpoints, [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0])

    def test_retry_checkpoints_not_used_for_no_female_case(self) -> None:
        checkpoints = build_uncertain_retry_checkpoints(
            seed_hits=0,
            duration_sec=100.0,
            initial_scan_end_sec=20.0,
        )
        self.assertEqual(checkpoints, [])

    def test_retry_checkpoints_can_run_without_seed_hits_when_forced(self) -> None:
        checkpoints = build_uncertain_retry_checkpoints(
            seed_hits=0,
            duration_sec=100.0,
            initial_scan_end_sec=20.0,
            require_seed_hits=False,
        )
        self.assertEqual(checkpoints, [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0])

    def test_retry_checkpoints_skip_already_scanned_ranges(self) -> None:
        checkpoints = build_uncertain_retry_checkpoints(
            seed_hits=3,
            duration_sec=100.0,
            initial_scan_end_sec=50.0,
        )
        self.assertEqual(checkpoints, [55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0, 95.0])

    def test_retry_sequence_stops_after_first_success(self) -> None:
        calls: list[tuple[float, float]] = []
        cfg = Config(input_dir=".", output_dir=".")
        checkpoints = [25.0, 50.0, 75.0]

        def fake_scan(**kwargs):
            scan_start = float(kwargs.get("scan_start_sec", 0.0))
            scan_end = float(kwargs.get("scan_duration", 0.0))
            calls.append((scan_start, scan_end))
            if scan_start == 50.0:
                return (
                    52.0,
                    (10, 10, 60, 60),
                    object(),
                    "Female",
                    0.91,
                    {
                        "female_seed_hits": 1,
                        "best_seed_confidence": 0.91,
                        "total_faces_evaluated": 8,
                        "low_face_area_rejections": 1,
                    },
                )
            return (
                None,
                None,
                None,
                None,
                None,
                {
                    "female_seed_hits": 1,
                    "best_seed_confidence": 0.72,
                    "total_faces_evaluated": 5,
                    "low_face_area_rejections": 2,
                },
            )

        (
            first_ts,
            first_box,
            first_embedding,
            first_gender_label,
            first_gender_conf,
            scan_info,
            retry_scan_end,
            attempted_checkpoints,
        ) = retry_find_first_female(
            cap=None,  # type: ignore[arg-type]
            fps=30.0,
            total_frames=3000,
            duration_sec=100.0,
            scan_window_sec=20.0,
            cfg=cfg,
            video_name="sample.mp4",
            checkpoint_starts=checkpoints,
            scan_callable=fake_scan,
        )

        self.assertEqual(calls, [(25.0, 45.0), (50.0, 70.0)])
        self.assertEqual(attempted_checkpoints, [25.0, 50.0])
        self.assertEqual(first_ts, 52.0)
        self.assertEqual(first_box, (10, 10, 60, 60))
        self.assertIsNotNone(first_embedding)
        self.assertEqual(first_gender_label, "Female")
        self.assertEqual(first_gender_conf, 0.91)
        self.assertEqual(retry_scan_end, 70.0)
        self.assertEqual(scan_info["female_seed_hits"], 2)
        self.assertEqual(scan_info["total_faces_evaluated"], 13)
        self.assertEqual(scan_info["low_face_area_rejections"], 3)
        self.assertEqual(scan_info["best_seed_confidence"], 0.91)


if __name__ == "__main__":
    unittest.main()
