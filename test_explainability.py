import unittest

from sort_videos_by_female_faces_gpu import (
    Config,
    apply_explainability_metadata,
    build_result_json_payload,
    new_result_record,
)


class ExplainabilityTests(unittest.TestCase):
    def _cfg(self) -> Config:
        return Config(input_dir=".", output_dir=".")

    def test_low_face_area_and_unverified_candidates_tags(self) -> None:
        cfg = self._cfg()
        result = new_result_record("a.mp4", "CPU fallback")
        result["decision_label"] = "uncertain"
        result["decision_reason"] = "Female-like candidates found (3) but none verified across confirmation frames."
        result["reason_metrics"] = {
            "total_faces_evaluated": 10,
            "low_face_area_rejections": 8,
            "female_seed_hits": 3,
        }

        apply_explainability_metadata(result, cfg)
        tags = result["reason_tags"]
        self.assertIn("low_face_area", tags)
        self.assertIn("unverified_female_candidates", tags)
        self.assertEqual(result["reason_summary"], "Most candidate faces were too small for reliable evidence.")

    def test_gender_disagreement_priority_over_few_votes(self) -> None:
        cfg = self._cfg()
        result = new_result_record("b.mp4", "CPU fallback")
        result["decision_label"] = "uncertain"
        result["decision_reason"] = "Conflicting gender votes during stabilization (female_score=0.620, male_score=0.910)."
        result["reason_metrics"] = {
            "gender_votes": 2,
            "female_score": 0.62,
            "male_score": 0.91,
            "stable_embeddings": 4,
        }

        apply_explainability_metadata(result, cfg)
        tags = result["reason_tags"]
        self.assertIn("few_votes", tags)
        self.assertIn("gender_disagreement", tags)
        self.assertEqual(result["reason_summary"], "Gender evidence disagreed during stabilization.")

    def test_result_json_payload_shape(self) -> None:
        cfg = self._cfg()
        result = new_result_record("c.mp4", "CPU fallback")
        result["decision_label"] = "female_detected"
        result["decision_reason"] = "Consistent female detection and stabilized identity track."
        result["confidence_score"] = 0.87654
        result["reason_metrics"] = {
            "stable_embeddings": 5,
            "gender_votes": 4,
            "female_score": 2.2,
            "male_score": 0.1,
        }
        apply_explainability_metadata(result, cfg)

        payload = build_result_json_payload(result)
        self.assertIn("video", payload)
        self.assertIn("video_name", payload)
        self.assertIn("decision_label", payload)
        self.assertIn("confidence_score", payload)
        self.assertIn("reason_summary", payload)
        self.assertIn("reason_tags", payload)
        self.assertIn("reason_metrics", payload)
        self.assertIn("embedding_source", payload)
        self.assertEqual(payload["video_name"], "c.mp4")
        self.assertEqual(payload["confidence_score"], 0.877)
        self.assertEqual(payload["embedding_source"], "")
        self.assertIsInstance(payload["reason_tags"], list)
        self.assertIsInstance(payload["reason_metrics"], dict)


if __name__ == "__main__":
    unittest.main()
