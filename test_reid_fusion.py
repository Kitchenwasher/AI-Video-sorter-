import unittest

from reid_fusion import rerank_similarity


class ReidFusionTests(unittest.TestCase):
    def test_direct_accept_without_reid_usage(self) -> None:
        decision = rerank_similarity(
            arcface_similarity=0.92,
            threshold=0.80,
            reid_similarity=None,
            reid_enabled=True,
            reid_fusion_weight=0.35,
            reid_min_similarity=0.55,
            reid_ambiguity_margin_low=0.08,
            reid_ambiguity_margin_high=0.06,
        )
        self.assertTrue(decision["accepted"])
        self.assertFalse(decision["reid_used"])
        self.assertEqual(decision["mode"], "direct_accept")

    def test_ambiguous_fused_accept(self) -> None:
        decision = rerank_similarity(
            arcface_similarity=0.79,
            threshold=0.80,
            reid_similarity=0.95,
            reid_enabled=True,
            reid_fusion_weight=0.35,
            reid_min_similarity=0.55,
            reid_ambiguity_margin_low=0.08,
            reid_ambiguity_margin_high=0.06,
        )
        self.assertTrue(decision["accepted"])
        self.assertTrue(decision["reid_used"])
        self.assertEqual(decision["mode"], "ambiguous_fused")
        self.assertGreaterEqual(float(decision["fused_score"]), 0.80)

    def test_ambiguous_fused_reject_on_low_reid_score(self) -> None:
        decision = rerank_similarity(
            arcface_similarity=0.81,
            threshold=0.80,
            reid_similarity=0.30,
            reid_enabled=True,
            reid_fusion_weight=0.35,
            reid_min_similarity=0.55,
            reid_ambiguity_margin_low=0.08,
            reid_ambiguity_margin_high=0.06,
        )
        self.assertFalse(decision["accepted"])
        self.assertTrue(decision["reid_used"])
        self.assertEqual(decision["mode"], "ambiguous_fused")

    def test_ambiguous_fallback_when_reid_missing(self) -> None:
        decision = rerank_similarity(
            arcface_similarity=0.81,
            threshold=0.80,
            reid_similarity=None,
            reid_enabled=True,
            reid_fusion_weight=0.35,
            reid_min_similarity=0.55,
            reid_ambiguity_margin_low=0.08,
            reid_ambiguity_margin_high=0.06,
        )
        self.assertTrue(decision["accepted"])
        self.assertFalse(decision["reid_used"])
        self.assertEqual(decision["mode"], "ambiguous_arcface_fallback")


if __name__ == "__main__":
    unittest.main()

