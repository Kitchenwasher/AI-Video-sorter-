import csv
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sort_videos_by_female_faces_gpu import (
    build_per_video_report_rows,
    build_run_summary_payload,
    report_output_dir,
    write_run_reports,
)


class SummaryReportsTests(unittest.TestCase):
    def test_build_run_summary_payload(self) -> None:
        start = datetime(2026, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        finish = start + timedelta(seconds=12.3456)
        payload = build_run_summary_payload(
            run_id="20260101T100000Z-abc12345",
            run_started_at=start,
            run_finished_at=finish,
            total_scanned=20,
            processed_successfully=18,
            female_detected=9,
            uncertain=4,
            no_female_found=5,
            errors=2,
            stopped_early=False,
        )
        self.assertEqual(payload["run_id"], "20260101T100000Z-abc12345")
        self.assertEqual(payload["total_scanned"], 20)
        self.assertEqual(payload["female_detected"], 9)
        self.assertEqual(payload["uncertain"], 4)
        self.assertEqual(payload["errors"], 2)
        self.assertAlmostEqual(float(payload["time_taken_seconds"]), 12.346, places=3)

    def test_build_per_video_report_rows(self) -> None:
        rows = build_per_video_report_rows(
            [
                {
                    "video": "C:/in/a.mp4",
                    "decision_label": "female_detected",
                    "confidence_score": 0.91234,
                    "reason_summary": "Stable evidence",
                    "error": "",
                    "memory_applied": True,
                    "final_destination": "C:/out/Female_A/a.mp4",
                },
                {
                    "video": "C:/in/b.mp4",
                    "decision_label": "error",
                    "error": "decode failed",
                },
            ]
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["video_name"], "a.mp4")
        self.assertEqual(rows[0]["confidence_score"], 0.912)
        self.assertEqual(rows[1]["decision_label"], "error")
        self.assertEqual(rows[1]["error"], "decode failed")

    def test_write_run_reports_creates_json_and_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_id = "20260101T100000Z-abc12345"
            summary = {
                "run_id": run_id,
                "run_started_at": "2026-01-01T10:00:00+00:00",
                "run_finished_at": "2026-01-01T10:00:12+00:00",
                "time_taken_seconds": 12.0,
                "total_scanned": 10,
                "processed_successfully": 9,
                "female_detected": 4,
                "uncertain": 2,
                "no_female_found": 3,
                "errors": 1,
                "stopped_early": False,
            }
            videos = [
                {
                    "video_path": "C:/in/a.mp4",
                    "video_name": "a.mp4",
                    "decision_label": "female_detected",
                    "confidence_score": 0.9,
                    "reason_summary": "stable",
                    "error": "",
                    "memory_applied": True,
                    "final_destination": "C:/out/Female_A/a.mp4",
                }
            ]

            paths = write_run_reports(output_dir=output_dir, run_id=run_id, summary_payload=summary, video_rows=videos)
            for key in ("summary_json", "summary_csv", "videos_json", "videos_csv"):
                self.assertTrue(Path(paths[key]).exists(), key)

            reports_dir = report_output_dir(output_dir)
            self.assertFalse(any(p.suffix == ".tmp" for p in reports_dir.iterdir()))

            summary_json = json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8"))
            self.assertEqual(summary_json["total_scanned"], 10)

            with Path(paths["summary_csv"]).open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["errors"], "1")

            videos_json = json.loads(Path(paths["videos_json"]).read_text(encoding="utf-8"))
            self.assertEqual(len(videos_json), 1)
            self.assertEqual(videos_json[0]["video_name"], "a.mp4")


if __name__ == "__main__":
    unittest.main()
