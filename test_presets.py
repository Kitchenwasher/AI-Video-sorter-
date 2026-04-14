import argparse
import unittest

from sort_videos_by_female_faces_gpu import (
    PROFILE_BALANCED,
    PROFILE_HIGH_ACCURACY,
    apply_profile_defaults,
    apply_uncertain_reprocess_defaults,
    normalize_profile_name,
)


class PresetTests(unittest.TestCase):
    def test_normalize_profile_aliases(self) -> None:
        self.assertEqual(normalize_profile_name("High Accuracy"), "high_accuracy")
        self.assertEqual(normalize_profile_name("high-accuracy"), "high_accuracy")
        self.assertEqual(normalize_profile_name("FAST"), "fast")
        self.assertEqual(normalize_profile_name("unknown"), PROFILE_BALANCED)

    def test_apply_profile_defaults_fast(self) -> None:
        args = argparse.Namespace(
            profile="fast",
            max_seconds=60,
            sample_every_sec=2.0,
            stabilization_seconds=8.0,
            resize_width=960,
            detection_batch_size=4,
            female_confirmation_frames=2,
            min_female_vote_ratio=0.62,
            min_stable_embeddings=3,
            min_stabilization_gender_votes=3,
            same_person_threshold=0.72,
            cluster_merge_threshold=0.78,
        )
        selected = apply_profile_defaults(args)
        self.assertEqual(selected, "fast")
        self.assertEqual(args.max_seconds, 40)
        self.assertEqual(args.resize_width, 720)
        self.assertEqual(args.min_stable_embeddings, 2)

    def test_profile_respects_explicit_manual_flag(self) -> None:
        args = argparse.Namespace(
            profile="fast",
            max_seconds=123,
            sample_every_sec=2.0,
            stabilization_seconds=8.0,
            resize_width=960,
            detection_batch_size=4,
            female_confirmation_frames=2,
            min_female_vote_ratio=0.62,
            min_stable_embeddings=3,
            min_stabilization_gender_votes=3,
            same_person_threshold=0.72,
            cluster_merge_threshold=0.78,
        )
        apply_profile_defaults(args, ["--max-seconds"])
        self.assertEqual(args.max_seconds, 123)
        self.assertEqual(args.resize_width, 720)

    def test_uncertain_reprocess_defaults_use_high_accuracy_base(self) -> None:
        args = argparse.Namespace(
            profile="balanced",
            max_seconds=60,
            sample_every_sec=2.0,
            stabilization_seconds=8.0,
            resize_width=960,
            detection_batch_size=4,
            female_confirmation_frames=2,
            min_female_vote_ratio=0.62,
            min_stable_embeddings=3,
            min_stabilization_gender_votes=3,
            same_person_threshold=0.72,
            cluster_merge_threshold=0.78,
        )
        selected = apply_uncertain_reprocess_defaults(args)
        self.assertEqual(selected, PROFILE_HIGH_ACCURACY)
        self.assertEqual(args.profile, PROFILE_HIGH_ACCURACY)
        self.assertEqual(args.max_seconds, 90)
        self.assertEqual(args.female_confirmation_frames, 4)
        self.assertEqual(args.min_stable_embeddings, 5)
        self.assertEqual(args.min_stabilization_gender_votes, 5)
        self.assertEqual(args.min_female_vote_ratio, 0.70)

    def test_uncertain_reprocess_respects_manual_threshold_overrides(self) -> None:
        args = argparse.Namespace(
            profile="balanced",
            max_seconds=60,
            sample_every_sec=2.0,
            stabilization_seconds=8.0,
            resize_width=960,
            detection_batch_size=4,
            female_confirmation_frames=2,
            min_female_vote_ratio=0.62,
            min_stable_embeddings=3,
            min_stabilization_gender_votes=3,
            same_person_threshold=0.72,
            cluster_merge_threshold=0.78,
        )
        apply_uncertain_reprocess_defaults(
            args,
            ["--female-confirmation-frames", "--min-female-vote-ratio"],
        )
        self.assertEqual(args.female_confirmation_frames, 2)
        self.assertEqual(args.min_female_vote_ratio, 0.62)
        self.assertEqual(args.min_stable_embeddings, 5)


if __name__ == "__main__":
    unittest.main()
