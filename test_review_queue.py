import tempfile
import unittest
from pathlib import Path

from review_queue import (
    apply_review_action,
    create_review_state,
    decide_review_route,
    load_review_state,
    save_review_state,
)


class ReviewQueueTests(unittest.TestCase):
    def test_queue_routing_review_mode(self) -> None:
        self.assertEqual(decide_review_route({"decision_label": "female_detected"}, review_mode=True), "queue")
        self.assertEqual(decide_review_route({"decision_label": "uncertain"}, review_mode=True), "queue")
        self.assertEqual(decide_review_route({"decision_label": "no_female"}, review_mode=True), "direct_no_female")
        self.assertEqual(decide_review_route({"decision_label": "female_detected"}, review_mode=False), "direct")

    def test_review_state_atomic_skip_transition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            queue_items = [
                {
                    "id": "item-1",
                    "source_path": str(output_dir / "src.mp4"),
                    "pending_path": str(output_dir / "Review_Pending" / "Needs_Review" / "src.mp4"),
                    "predicted_label": "uncertain",
                    "reason": "low evidence",
                    "confidence": 0.42,
                    "suggested_folder": "Needs_Review",
                    "status": "pending",
                    "final_path": "",
                    "reviewed_at": "",
                }
            ]
            state = create_review_state(output_dir, queue_items, run_id="run-test")
            save_review_state(output_dir, state)

            updated = apply_review_action(output_dir, "item-1", "skip")
            self.assertEqual(updated["status"], "skipped")
            self.assertTrue(updated["reviewed_at"])
            self.assertTrue(updated["final_path"].endswith("src.mp4"))

            reloaded = load_review_state(output_dir)
            self.assertEqual(reloaded["schema_version"], 1)
            self.assertEqual(reloaded["items"][0]["status"], "skipped")

    def test_reassign_move_and_sanitize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            pending_root = output_dir / "Review_Pending" / "Female_Alice"
            pending_root.mkdir(parents=True, exist_ok=True)
            pending_video = pending_root / "clip.mp4"
            pending_video.write_bytes(b"video")

            existing_dir = output_dir / "Female_Alice"
            existing_dir.mkdir(parents=True, exist_ok=True)
            (existing_dir / "clip.mp4").write_bytes(b"existing")

            queue_items = [
                {
                    "id": "item-2",
                    "source_path": str(output_dir / "src2.mp4"),
                    "pending_path": str(pending_video),
                    "predicted_label": "female_detected",
                    "reason": "female stable",
                    "confidence": 0.83,
                    "suggested_folder": "Female_Alice",
                    "status": "pending",
                    "final_path": "",
                    "reviewed_at": "",
                }
            ]
            save_review_state(output_dir, create_review_state(output_dir, queue_items, run_id="run-test-2"))

            updated = apply_review_action(output_dir, "item-2", "reassign_existing", "Female_Alice")
            moved_path = Path(updated["final_path"])
            self.assertTrue(moved_path.exists())
            self.assertTrue(moved_path.name.startswith("clip"))
            self.assertNotEqual(moved_path.name, "clip.mp4")

            pending_two = pending_root / "clip2.mp4"
            pending_two.write_bytes(b"video2")
            queue_items_two = [
                {
                    "id": "item-3",
                    "source_path": str(output_dir / "src3.mp4"),
                    "pending_path": str(pending_two),
                    "predicted_label": "uncertain",
                    "reason": "manual review",
                    "confidence": 0.5,
                    "suggested_folder": "Needs_Review",
                    "status": "pending",
                    "final_path": "",
                    "reviewed_at": "",
                }
            ]
            save_review_state(output_dir, create_review_state(output_dir, queue_items_two, run_id="run-test-3"))

            updated_two = apply_review_action(output_dir, "item-3", "reassign_new", "Jane:*?Folder")
            moved_path_two = Path(updated_two["final_path"])
            self.assertTrue(moved_path_two.exists())
            self.assertIn("Jane", str(moved_path_two.parent.name))


if __name__ == "__main__":
    unittest.main()
