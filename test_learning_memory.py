import json
import tempfile
import unittest
from pathlib import Path

from learning_memory import (
    build_learning_summary,
    compute_adaptive_auto_threshold,
    default_memory_path,
    load_memory,
    match_identity,
    record_feedback,
    save_memory,
)


class LearningMemoryTests(unittest.TestCase):
    def test_load_default_and_corrupt_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            mem_path = default_memory_path(base)
            memory = load_memory(mem_path)
            self.assertEqual(memory["schema_version"], 1)
            self.assertEqual(memory["identities"], [])

            mem_path.parent.mkdir(parents=True, exist_ok=True)
            mem_path.write_text("{not-json", encoding="utf-8")
            recovered = load_memory(mem_path)
            self.assertEqual(recovered["schema_version"], 1)
            self.assertEqual(recovered["identities"], [])

    def test_identity_lock_defaults_on_legacy_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            mem_path = default_memory_path(base)
            mem_path.parent.mkdir(parents=True, exist_ok=True)
            mem_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "identities": [
                            {
                                "label": "Female_Old",
                                "prototype": [1.0, 0.0, 0.0],
                                "sample_count": 1,
                                "confidence_sum": 0.9,
                            }
                        ],
                        "decisions": [],
                        "stats": {},
                    }
                ),
                encoding="utf-8",
            )
            loaded = load_memory(mem_path)
            self.assertEqual(len(loaded["identities"]), 1)
            identity = loaded["identities"][0]
            self.assertFalse(identity.get("locked", True))
            self.assertEqual(identity.get("locked_at", ""), "")
            self.assertEqual(int(identity.get("positive_feedback_count", 0)), 0)
            self.assertEqual(int(identity.get("negative_feedback_count", 0)), 0)
            self.assertIn("adaptive_auto_threshold", identity)
            self.assertIn("correction_consistency_score", identity)

    def test_match_identity_and_threshold_data(self) -> None:
        memory = {
            "schema_version": 1,
            "updated_at": "",
            "identities": [
                {
                    "label": "Female_Alice",
                    "prototype": [1.0, 0.0, 0.0],
                    "sample_count": 1,
                    "confidence_sum": 1.0,
                    "last_used": "",
                },
                {
                    "label": "Female_Jane",
                    "prototype": [0.0, 1.0, 0.0],
                    "sample_count": 1,
                    "confidence_sum": 1.0,
                    "last_used": "",
                },
            ],
            "decisions": [],
            "stats": {},
        }

        match = match_identity(memory, [0.9, 0.1, 0.0])
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match["label"], "Female_Alice")
        self.assertGreater(match["score"], 0.8)
        self.assertIn("adaptive_auto_threshold", match)
        self.assertIn("correction_consistency_score", match)

    def test_record_feedback_identity_and_no_female(self) -> None:
        memory = load_memory(Path("does-not-matter.json"))

        record_feedback(
            memory,
            action="approve_suggested",
            label="Female_Alice",
            predicted_label="female_detected",
            confidence=0.88,
            source_path="src1.mp4",
            final_path="out/Female_Alice/src1.mp4",
            embedding=[1.0, 0.0, 0.0],
        )
        self.assertEqual(len(memory["identities"]), 1)
        self.assertEqual(memory["identities"][0]["label"], "Female_Alice")
        self.assertEqual(memory["stats"]["total_identity_updates"], 1)
        self.assertEqual(memory["stats"]["total_positive_events"], 1)
        self.assertEqual(memory["identities"][0]["positive_feedback_count"], 1)

        record_feedback(
            memory,
            action="move_no_female",
            label="No_Female_Found",
            predicted_label="no_female",
            confidence=0.9,
            source_path="src2.mp4",
            final_path="out/No_Female_Found/src2.mp4",
            embedding=None,
            feedback_event_type="negative",
            negative_label="Female_Alice",
        )
        self.assertEqual(memory["stats"]["total_no_female_events"], 1)
        self.assertEqual(memory["stats"]["total_feedback_events"], 2)
        self.assertEqual(memory["stats"]["total_negative_events"], 1)
        identity = next(item for item in memory["identities"] if item["label"] == "Female_Alice")
        self.assertEqual(identity["negative_feedback_count"], 1)

    def test_save_and_reload_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            mem_path = default_memory_path(base)
            memory = load_memory(mem_path)
            record_feedback(
                memory,
                action="reassign_new",
                label="Female_Test",
                predicted_label="uncertain",
                confidence=0.77,
                source_path="a.mp4",
                final_path="b.mp4",
                embedding=[0.5, 0.5, 0.0],
            )
            save_memory(mem_path, memory)
            loaded = json.loads(mem_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["schema_version"], 1)
            self.assertEqual(len(loaded["identities"]), 1)
            self.assertEqual(loaded["identities"][0]["label"], "Female_Test")

    def test_adaptive_threshold_bounds(self) -> None:
        lower = compute_adaptive_auto_threshold(
            global_auto_threshold=0.82,
            global_suggest_threshold=0.74,
            positive_feedback_count=30,
            negative_feedback_count=0,
        )
        self.assertGreaterEqual(lower, 0.74)
        self.assertLessEqual(lower, 0.82)

        upper = compute_adaptive_auto_threshold(
            global_auto_threshold=0.82,
            global_suggest_threshold=0.74,
            positive_feedback_count=0,
            negative_feedback_count=30,
        )
        self.assertGreaterEqual(upper, 0.74)
        self.assertLessEqual(upper, 0.82)
        self.assertGreaterEqual(upper, lower)

    def test_learning_summary_shape(self) -> None:
        memory = load_memory(Path("does-not-matter-2.json"))
        record_feedback(
            memory,
            action="approve_suggested",
            source_action="review_approve_suggested",
            feedback_event_type="positive",
            label="Female_Summary",
            predicted_label="female_detected",
            confidence=0.91,
            source_path="a.mp4",
            final_path="out/Female_Summary/a.mp4",
            embedding=[1.0, 0.0, 0.0],
        )
        summary = build_learning_summary(memory)
        self.assertTrue(summary["ok"])
        self.assertGreaterEqual(summary["count"], 1)
        first = summary["items"][0]
        self.assertIn("adaptive_auto_threshold", first)
        self.assertIn("recent_trend", first)


if __name__ == "__main__":
    unittest.main()
