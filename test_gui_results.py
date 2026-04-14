import json
import unittest

from video_sorter_gui import RESULT_JSON_PREFIX, format_result_row, parse_result_json_line


class GuiResultsTests(unittest.TestCase):
    def test_parse_result_json_line_valid(self) -> None:
        payload = {
            "video": "/tmp/clip.mp4",
            "video_name": "clip.mp4",
            "decision_label": "uncertain",
            "confidence_score": 0.4321,
            "reason_summary": "Too few votes.",
            "reason_tags": ["few_votes"],
        }
        line = f"{RESULT_JSON_PREFIX}{json.dumps(payload)}"
        parsed = parse_result_json_line(line)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["video_name"], "clip.mp4")
        self.assertEqual(parsed["decision_label"], "uncertain")

    def test_parse_result_json_line_invalid(self) -> None:
        self.assertIsNone(parse_result_json_line("normal log line"))
        self.assertIsNone(parse_result_json_line(f"{RESULT_JSON_PREFIX}not-json"))

    def test_format_result_row(self) -> None:
        payload = {
            "video": "/tmp/sample.mp4",
            "decision_label": "female_detected",
            "confidence_score": 0.9,
            "reason_summary": "Stable and consistent evidence.",
            "reason_tags": ["few_stable_embeddings", "memory_match_suggested"],
        }
        row = format_result_row(payload)
        self.assertEqual(row[0], "sample.mp4")
        self.assertEqual(row[1], "female_detected")
        self.assertEqual(row[2], "0.900")
        self.assertEqual(row[3], "Stable and consistent evidence.")
        self.assertEqual(row[4], "few_stable_embeddings, memory_match_suggested")


if __name__ == "__main__":
    unittest.main()
