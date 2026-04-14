import json
import os
import tempfile
import unittest
from pathlib import Path

from identity_tools import list_identities, perform_identity_action
from learning_memory import default_memory_path, load_memory, save_memory


class IdentityToolsTests(unittest.TestCase):
    def test_list_identities_excludes_system_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "Female_A").mkdir(parents=True, exist_ok=True)
            (output_dir / "Female_A" / "a.mp4").write_bytes(b"a")
            (output_dir / "No_Female_Found").mkdir(parents=True, exist_ok=True)
            (output_dir / "No_Female_Found" / "b.mp4").write_bytes(b"b")
            (output_dir / "Review_Pending").mkdir(parents=True, exist_ok=True)

            mem_path = default_memory_path(output_dir)
            memory = load_memory(mem_path)
            memory["identities"] = [
                {
                    "label": "Female_A",
                    "prototype": [1.0, 0.0, 0.0],
                    "sample_count": 2,
                    "confidence_sum": 1.8,
                    "last_used": "2026-04-14T10:00:00+00:00",
                    "locked": True,
                    "locked_at": "2026-04-14T10:00:00+00:00",
                }
            ]
            save_memory(mem_path, memory)

            payload = list_identities(output_dir, load_memory(mem_path))
            self.assertTrue(payload["ok"])
            self.assertEqual(payload["count"], 1)
            self.assertEqual(payload["identities"][0]["name"], "Female_A")
            self.assertTrue(payload["identities"][0]["locked"])

    def test_merge_updates_files_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            source_dir = output_dir / "Female_A"
            target_dir = output_dir / "Female_B"
            source_dir.mkdir(parents=True, exist_ok=True)
            target_dir.mkdir(parents=True, exist_ok=True)
            (source_dir / "clip.mp4").write_bytes(b"clip-a")
            (source_dir / "clip2.mp4").write_bytes(b"clip-a2")
            (target_dir / "clip.mp4").write_bytes(b"clip-b")

            mem_path = default_memory_path(output_dir)
            memory = load_memory(mem_path)
            memory["identities"] = [
                {
                    "label": "Female_A",
                    "prototype": [1.0, 0.0, 0.0],
                    "sample_count": 2,
                    "confidence_sum": 1.5,
                    "last_used": "2026-04-12T10:00:00+00:00",
                    "locked": True,
                    "locked_at": "2026-04-12T10:00:00+00:00",
                },
                {
                    "label": "Female_B",
                    "prototype": [0.0, 1.0, 0.0],
                    "sample_count": 1,
                    "confidence_sum": 0.7,
                    "last_used": "2026-04-11T10:00:00+00:00",
                    "locked": False,
                    "locked_at": "",
                },
            ]
            memory["decisions"] = [
                {
                    "timestamp": "2026-04-13T10:00:00+00:00",
                    "action": "approve_suggested",
                    "label": "Female_A",
                    "predicted_label": "female_detected",
                    "confidence": 0.8,
                    "memory_match_label": "Female_A",
                    "memory_match_score": 0.9,
                    "source_path": "src.mp4",
                    "final_path": str(source_dir / "clip.mp4"),
                    "embedding_present": True,
                }
            ]
            save_memory(mem_path, memory)

            result = perform_identity_action(
                output_dir=output_dir,
                memory_path=mem_path,
                action="merge",
                source_folder="Female_A",
                target_folder="Female_B",
            )
            self.assertTrue(result["ok"])
            self.assertEqual(result["action"], "merge")
            self.assertEqual(result["details"]["moved_count"], 2)

            self.assertFalse(source_dir.exists())
            target_videos = sorted(p.name for p in target_dir.iterdir() if p.is_file())
            self.assertEqual(len(target_videos), 3)

            updated = load_memory(mem_path)
            labels = [str(item.get("label", "")) for item in updated.get("identities", [])]
            self.assertIn("Female_B", labels)
            self.assertNotIn("Female_A", labels)
            merged_identity = next(item for item in updated["identities"] if item["label"] == "Female_B")
            self.assertEqual(int(merged_identity.get("sample_count", 0)), 3)
            self.assertTrue(bool(merged_identity.get("locked", False)))

            decision = updated["decisions"][0]
            self.assertEqual(decision["label"], "Female_B")
            self.assertEqual(decision["memory_match_label"], "Female_B")
            self.assertIn("Female_B", str(decision.get("final_path", "")))
            self.assertTrue(
                any(
                    str(item.get("feedback_event_type", "")) == "structural"
                    and str(item.get("source_action", "")) == "identity_merge"
                    for item in updated["decisions"]
                )
            )

    def test_split_updates_decision_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            source_dir = output_dir / "Female_B"
            source_dir.mkdir(parents=True, exist_ok=True)
            x_file = source_dir / "x.mp4"
            y_file = source_dir / "y.mp4"
            x_file.write_bytes(b"x")
            y_file.write_bytes(b"y")

            mem_path = default_memory_path(output_dir)
            memory = load_memory(mem_path)
            memory["decisions"] = [
                {
                    "timestamp": "2026-04-13T10:00:00+00:00",
                    "action": "approve_suggested",
                    "label": "Female_B",
                    "predicted_label": "female_detected",
                    "confidence": 0.8,
                    "memory_match_label": "",
                    "memory_match_score": 0.0,
                    "source_path": "src.mp4",
                    "final_path": str(x_file.resolve()),
                    "embedding_present": True,
                }
            ]
            save_memory(mem_path, memory)

            result = perform_identity_action(
                output_dir=output_dir,
                memory_path=mem_path,
                action="split",
                source_folder="Female_B",
                target_folder="Female_C",
                selected_videos=[str(x_file)],
            )
            self.assertTrue(result["ok"])
            self.assertEqual(result["details"]["moved_count"], 1)
            self.assertTrue((output_dir / "Female_C").is_dir())
            self.assertTrue((output_dir / "Female_C" / "x.mp4").exists())
            self.assertTrue((source_dir / "y.mp4").exists())

            updated = load_memory(mem_path)
            decision = updated["decisions"][0]
            self.assertEqual(decision["label"], "Female_C")
            self.assertIn("Female_C", str(decision.get("final_path", "")))
            self.assertTrue(any(d.get("action") == "identity_split" for d in updated["decisions"]))
            self.assertTrue(
                any(
                    str(item.get("feedback_event_type", "")) == "structural"
                    and str(item.get("source_action", "")) == "identity_split"
                    for item in updated["decisions"]
                )
            )

    def test_merge_uses_embedding_cache_for_learning_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            source_dir = output_dir / "Female_A"
            target_dir = output_dir / "Female_B"
            source_dir.mkdir(parents=True, exist_ok=True)
            target_dir.mkdir(parents=True, exist_ok=True)
            clip = source_dir / "clip.mp4"
            clip.write_bytes(b"clip-a")

            cache_path = output_dir / ".learning" / "video_embedding_cache.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_payload = {
                "schema_version": 1,
                "updated_at": "2026-04-14T10:00:00+00:00",
                "entries": {
                    os.path.normcase(os.path.normpath(str(clip.resolve()))): {
                        "video_path": str(clip.resolve()),
                        "source_video_path": str(clip.resolve()),
                        "decision_label": "female_detected",
                        "confidence_score": 0.91,
                        "embedding": [1.0, 0.0, 0.0],
                        "updated_at": "2026-04-14T10:00:00+00:00",
                    }
                },
            }
            cache_path.write_text(json.dumps(cache_payload, indent=2), encoding="utf-8")

            mem_path = default_memory_path(output_dir)
            result = perform_identity_action(
                output_dir=output_dir,
                memory_path=mem_path,
                action="merge",
                source_folder="Female_A",
                target_folder="Female_B",
            )
            self.assertTrue(result["ok"])
            self.assertEqual(result["details"]["learning_feedback_with_embeddings"], 1)
            self.assertEqual(result["details"]["learning_feedback_events"], 2)

            updated = load_memory(mem_path)
            self.assertTrue(
                any(
                    str(item.get("source_action", "")) == "identity_merge"
                    and str(item.get("feedback_event_type", "")) == "positive"
                    for item in updated.get("decisions", [])
                )
            )
            self.assertTrue(
                any(
                    str(item.get("source_action", "")) == "identity_merge"
                    and str(item.get("feedback_event_type", "")) == "negative"
                    for item in updated.get("decisions", [])
                )
            )

    def test_lock_and_unlock_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            folder = output_dir / "Female_D"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "clip.mp4").write_bytes(b"d")
            mem_path = default_memory_path(output_dir)

            lock_result = perform_identity_action(
                output_dir=output_dir,
                memory_path=mem_path,
                action="lock",
                folder_name="Female_D",
            )
            self.assertTrue(lock_result["ok"])
            locked_memory = load_memory(mem_path)
            identity = next(item for item in locked_memory["identities"] if item["label"] == "Female_D")
            self.assertTrue(identity["locked"])
            self.assertTrue(str(identity.get("locked_at", "")))

            unlock_result = perform_identity_action(
                output_dir=output_dir,
                memory_path=mem_path,
                action="unlock",
                folder_name="Female_D",
            )
            self.assertTrue(unlock_result["ok"])
            unlocked_memory = load_memory(mem_path)
            identity_after = next(item for item in unlocked_memory["identities"] if item["label"] == "Female_D")
            self.assertFalse(identity_after["locked"])
            self.assertEqual(str(identity_after.get("locked_at", "")), "")
            self.assertTrue(
                any(str(item.get("source_action", "")) == "identity_lock" for item in unlocked_memory["decisions"])
            )
            self.assertTrue(
                any(str(item.get("source_action", "")) == "identity_unlock" for item in unlocked_memory["decisions"])
            )


if __name__ == "__main__":
    unittest.main()
