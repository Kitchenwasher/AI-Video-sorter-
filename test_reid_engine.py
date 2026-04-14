import unittest

import numpy as np

from reid_engine_torchreid import extract_reid_embedding, initialize_reid, reid_similarity


class ReidEngineTests(unittest.TestCase):
    def test_initialize_reid_is_safe(self) -> None:
        label = initialize_reid(model_tier="balanced", device_pref="cpu")
        self.assertIsInstance(label, str)
        self.assertGreater(len(label), 0)

    def test_extract_reid_embedding_handles_empty_input(self) -> None:
        emb = extract_reid_embedding(np.asarray([], dtype=np.uint8))
        self.assertIsNone(emb)

    def test_reid_similarity_bounds(self) -> None:
        a = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        sim = reid_similarity(a, b)
        self.assertGreaterEqual(sim, -1.0)
        self.assertLessEqual(sim, 1.0)
        self.assertAlmostEqual(reid_similarity(a, a), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()

