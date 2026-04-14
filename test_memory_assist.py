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


if __name__ == "__main__":
    unittest.main()
