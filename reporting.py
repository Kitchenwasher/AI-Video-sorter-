"""Reporting module – run summary payloads, per-video rows, CSV/JSON generation."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence

from shared.constants import REPORTS_DIRNAME
from shared.utils import atomic_write_text, clamp_confidence, ensure_dir, safe_float


# ---------------------------------------------------------------------------
# Report directory helpers.
# ---------------------------------------------------------------------------

def report_output_dir(output_dir: Path) -> Path:
    """Return the reports subdirectory inside *output_dir*."""
    return output_dir / REPORTS_DIRNAME


# ---------------------------------------------------------------------------
# Payload builders.
# ---------------------------------------------------------------------------

def build_run_summary_payload(
    *,
    run_id: str,
    run_started_at: datetime,
    run_finished_at: datetime,
    total_scanned: int,
    processed_successfully: int,
    female_detected: int,
    uncertain: int,
    no_female_found: int,
    errors: int,
    stopped_early: bool,
) -> Dict[str, Any]:
    time_taken_seconds = max(0.0, (run_finished_at - run_started_at).total_seconds())
    return {
        "run_id": run_id,
        "run_started_at": run_started_at.replace(microsecond=0).isoformat(),
        "run_finished_at": run_finished_at.replace(microsecond=0).isoformat(),
        "time_taken_seconds": round(time_taken_seconds, 3),
        "total_scanned": int(total_scanned),
        "processed_successfully": int(processed_successfully),
        "female_detected": int(female_detected),
        "uncertain": int(uncertain),
        "no_female_found": int(no_female_found),
        "errors": int(errors),
        "stopped_early": bool(stopped_early),
    }


def build_per_video_report_rows(results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in results:
        video_path = str(item.get("video", "")).strip()
        decision_label = str(item.get("decision_label", "")).strip().lower()
        rows.append(
            {
                "video_path": video_path,
                "video_name": Path(video_path).name if video_path else "",
                "decision_label": decision_label,
                "confidence_score": round(
                    clamp_confidence(safe_float(item.get("confidence_score", 0.0))), 3
                ),
                "reason_summary": str(item.get("reason_summary", "")).strip(),
                "error": str(item.get("error") or "").strip(),
                "memory_applied": bool(item.get("memory_applied", False)),
                "reid_enabled": bool(item.get("reid_enabled", False)),
                "reid_embedding_present": bool(item.get("reid_embedding")),
                "reid_backend": str(item.get("reid_backend", "")).strip(),
                "final_destination": str(item.get("final_destination", "")).strip(),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Report writing.
# ---------------------------------------------------------------------------

def write_run_reports(
    *,
    output_dir: Path,
    run_id: str,
    summary_payload: Dict[str, Any],
    video_rows: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    reports_dir = report_output_dir(output_dir)
    ensure_dir(reports_dir)

    summary_json_path = reports_dir / f"run_{run_id}_summary.json"
    summary_csv_path = reports_dir / f"run_{run_id}_summary.csv"
    videos_json_path = reports_dir / f"run_{run_id}_videos.json"
    videos_csv_path = reports_dir / f"run_{run_id}_videos.csv"

    atomic_write_text(
        summary_json_path,
        json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n",
    )
    atomic_write_text(
        videos_json_path,
        json.dumps(list(video_rows), indent=2, ensure_ascii=False) + "\n",
    )

    summary_headers = [
        "run_id",
        "run_started_at",
        "run_finished_at",
        "time_taken_seconds",
        "total_scanned",
        "processed_successfully",
        "female_detected",
        "uncertain",
        "no_female_found",
        "errors",
        "stopped_early",
    ]
    videos_headers = [
        "video_path",
        "video_name",
        "decision_label",
        "confidence_score",
        "reason_summary",
        "error",
        "memory_applied",
        "reid_enabled",
        "reid_embedding_present",
        "reid_backend",
        "final_destination",
    ]

    summary_io = io.StringIO()
    summary_writer = csv.DictWriter(summary_io, fieldnames=summary_headers, extrasaction="ignore")
    summary_writer.writeheader()
    summary_writer.writerow(summary_payload)
    atomic_write_text(summary_csv_path, summary_io.getvalue())

    videos_io = io.StringIO()
    videos_writer = csv.DictWriter(videos_io, fieldnames=videos_headers, extrasaction="ignore")
    videos_writer.writeheader()
    for row in video_rows:
        videos_writer.writerow(row)
    atomic_write_text(videos_csv_path, videos_io.getvalue())

    return {
        "summary_json": str(summary_json_path),
        "summary_csv": str(summary_csv_path),
        "videos_json": str(videos_json_path),
        "videos_csv": str(videos_csv_path),
    }


def derive_decision_counts(results: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Tally per-decision-label counts from a sequence of result dicts."""
    female_detected = 0
    uncertain = 0
    no_female_found = 0
    errors = 0
    processed_successfully = 0
    for item in results:
        has_error = bool(item.get("error"))
        stopped = bool(item.get("stopped"))
        decision = str(item.get("decision_label", "")).strip().lower()
        if has_error or decision == "error":
            errors += 1
            continue
        if stopped or decision == "stopped":
            continue
        processed_successfully += 1
        if decision == "female_detected":
            female_detected += 1
        elif decision == "uncertain":
            uncertain += 1
        elif decision == "no_female":
            no_female_found += 1
    return {
        "processed_successfully": processed_successfully,
        "female_detected": female_detected,
        "uncertain": uncertain,
        "no_female_found": no_female_found,
        "errors": errors,
    }


def emit_run_reports(
    *,
    output_dir: Path,
    run_id: str,
    run_started_at: datetime,
    run_finished_at: datetime,
    total_scanned: int,
    processed_successfully: int,
    female_detected: int,
    uncertain: int,
    no_female_found: int,
    errors: int,
    stopped_early: bool,
    results: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    summary_payload = build_run_summary_payload(
        run_id=run_id,
        run_started_at=run_started_at,
        run_finished_at=run_finished_at,
        total_scanned=total_scanned,
        processed_successfully=processed_successfully,
        female_detected=female_detected,
        uncertain=uncertain,
        no_female_found=no_female_found,
        errors=errors,
        stopped_early=stopped_early,
    )
    video_rows = build_per_video_report_rows(results)
    report_paths = write_run_reports(
        output_dir=output_dir,
        run_id=run_id,
        summary_payload=summary_payload,
        video_rows=video_rows,
    )
    print(
        "[REPORT] "
        f"summary_json={report_paths['summary_json']} "
        f"summary_csv={report_paths['summary_csv']} "
        f"videos_json={report_paths['videos_json']} "
        f"videos_csv={report_paths['videos_csv']}",
        flush=True,
    )
    return report_paths
