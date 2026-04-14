import unittest

from sort_videos_by_female_faces_gpu import Config, apply_memory_assist


class MemoryAssistTests(unittest.TestCase):
    def _cfg(self) -> Config:
        return Config(
            input_dir=".",
            output_dir=".",
            learning_enabled=True,
            learning_auto_threshold=0.82,
            learning_suggest_threshold=0.74,
        )

    def test_auto_apply_high_similarity(self) -> None:
        cfg = self._cfg()
        memory = {
            "identities": [
                {"label": "Female_Alice", "prototype": [1.0, 0.0, 0.0], "sample_count": 1, "confidence_sum": 1.0}
            ]
        }
        result = {
            "decision_label": "uncertain",
            "decision_reason": "low confidence",
            "confidence_score": 0.45,
            "embedding": [0.95, 0.05, 0.0],
            "female_found": False,
            "suggested_folder_name": "",
        }

        apply_memory_assist(result, memory, cfg)
        self.assertTrue(result["memory_applied"])
        self.assertEqual(result["decision_label"], "female_detected")
        self.assertEqual(result["suggested_folder_name"], "Female_Alice")

    def test_suggest_mid_similarity(self) -> None:
        cfg = self._cfg()
        memory = {
            "identities": [
                {"label": "Female_Jane", "prototype": [1.0, 0.0, 0.0], "sample_count": 1, "confidence_sum": 1.0}
            ]
        }
        result = {
            "decision_label": "no_female",
            "decision_reason": "none found",
            "confidence_score": 0.2,
            "embedding": [0.74, 0.67, 0.0],
            "female_found": False,
            "suggested_folder_name": "",
        }

        apply_memory_assist(result, memory, cfg)
        self.assertFalse(result["memory_applied"])
        self.assertEqual(result["decision_label"], "uncertain")
        self.assertEqual(result["suggested_folder_name"], "Female_Jane")

    def test_ignore_low_similarity(self) -> None:
        cfg = self._cfg()
        memory = {
            "identities": [
                {"label": "Female_Z", "prototype": [1.0, 0.0, 0.0], "sample_count": 1, "confidence_sum": 1.0}
            ]
        }
        result = {
            "decision_label": "no_female",
            "decision_reason": "none found",
            "confidence_score": 0.8,
            "embedding": [0.2, 0.98, 0.0],
            "female_found": False,
            "suggested_folder_name": "",
        }

        apply_memory_assist(result, memory, cfg)
        self.assertFalse(result["memory_applied"])
        self.assertEqual(result["decision_label"], "no_female")
        self.assertEqual(result["suggested_folder_name"], "")

    def test_mid_similarity_forces_uncertain_even_from_female_detected(self) -> None:
        cfg = self._cfg()
        memory = {
            "identities": [
                {"label": "Female_Luna", "prototype": [1.0, 0.0, 0.0], "sample_count": 1, "confidence_sum": 1.0}
            ]
        }
        result = {
            "decision_label": "female_detected",
            "decision_reason": "native model found female",
            "confidence_score": 0.79,
            "embedding": [0.76, 0.65, 0.0],
            "female_found": True,
            "suggested_folder_name": "",
            "suggested_cluster_id": 3,
        }

        apply_memory_assist(result, memory, cfg)
        self.assertFalse(result["memory_applied"])
        self.assertEqual(result["decision_label"], "uncertain")
        self.assertEqual(result["suggested_folder_name"], "Female_Luna")
        self.assertFalse(result["female_found"])
        self.assertIsNone(result["suggested_cluster_id"])

    def test_locked_identity_forces_auto_apply_at_threshold(self) -> None:
        cfg = self._cfg()
        memory = {
            "identities": [
                {
                    "label": "Female_Locked",
                    "prototype": [1.0, 0.0, 0.0],
                    "sample_count": 6,
                    "confidence_sum": 5.4,
                    "locked": True,
                    "locked_at": "2026-01-01T00:00:00+00:00",
                }
            ]
        }
        result = {
            "decision_label": "no_female",
            "decision_reason": "native pipeline says no female",
            "confidence_score": 0.95,
            "embedding": [0.96, 0.04, 0.0],
            "female_found": False,
            "suggested_folder_name": "",
            "suggested_cluster_id": 8,
        }

        apply_memory_assist(result, memory, cfg)
        self.assertTrue(result["memory_applied"])
        self.assertEqual(result["decision_label"], "female_detected")
        self.assertEqual(result["suggested_folder_name"], "Female_Locked")
        self.assertFalse("no_female" in result["decision_reason"])
        self.assertIsNone(result["suggested_cluster_id"])

    def test_adaptive_threshold_can_auto_apply_below_global_auto(self) -> None:
        cfg = self._cfg()
        memory = {
            "identities": [
                {
                    "label": "Female_Adaptive",
                    "prototype": [1.0, 0.0, 0.0],
                    "sample_count": 10,
                    "confidence_sum": 8.0,
                    "positive_feedback_count": 25,
                    "negative_feedback_count": 0,
                }
            ]
        }
        result = {
            "decision_label": "uncertain",
            "decision_reason": "native confidence low",
            "confidence_score": 0.35,
            "embedding": [0.78, 0.62, 0.0],
            "female_found": False,
            "suggested_folder_name": "",
            "suggested_cluster_id": 2,
        }

        apply_memory_assist(result, memory, cfg)
        self.assertTrue(result["memory_applied"])
        self.assertTrue(result["learning_applied"])
        self.assertEqual(result["decision_label"], "female_detected")
        self.assertLess(float(result["adaptive_threshold_used"]), cfg.learning_auto_threshold)

    def test_conflicting_feedback_keeps_threshold_conservative(self) -> None:
        cfg = self._cfg()
        memory = {
            "identities": [
                {
                    "label": "Female_Conflicting",
                    "prototype": [1.0, 0.0, 0.0],
                    "sample_count": 15,
                    "confidence_sum": 12.0,
                    "positive_feedback_count": 20,
                    "negative_feedback_count": 20,
                }
            ]
        }
        result = {
            "decision_label": "no_female",
            "decision_reason": "none found",
            "confidence_score": 0.30,
            "embedding": [0.78, 0.62, 0.0],
            "female_found": False,
            "suggested_folder_name": "",
        }

        apply_memory_assist(result, memory, cfg)
        self.assertFalse(result["memory_applied"])
        self.assertEqual(result["decision_label"], "uncertain")
        self.assertAlmostEqual(float(result["adaptive_threshold_used"]), cfg.learning_auto_threshold, places=3)

    def test_reid_ambiguous_match_auto_applies_when_enabled(self) -> None:
        cfg = self._cfg()
        cfg.cross_video_reid = True
        cfg.reid_fusion_weight = 0.35
        cfg.reid_min_similarity = 0.55
        cfg.reid_ambiguity_margin_low = 0.08
        cfg.reid_ambiguity_margin_high = 0.06
        memory = {
            "identities": [
                {
                    "label": "Female_ReID",
                    "prototype": [1.0, 0.0, 0.0],
                    "sample_count": 1,
                    "confidence_sum": 1.0,
                    "same_person_threshold": 0.80,
                    "reid_prototype": [1.0, 0.0, 0.0],
                    "reid_sample_count": 1,
                }
            ]
        }
        result = {
            "decision_label": "uncertain",
            "decision_reason": "native confidence low",
            "confidence_score": 0.30,
            "embedding": [0.79, 0.61, 0.0],
            "reid_embedding": [0.98, 0.02, 0.0],
            "female_found": False,
            "suggested_folder_name": "",
        }

        apply_memory_assist(result, memory, cfg)
        self.assertTrue(result["memory_applied"])
        self.assertEqual(result["decision_label"], "female_detected")
        self.assertEqual(result["suggested_folder_name"], "Female_ReID")


if __name__ == "__main__":
    unittest.main()
