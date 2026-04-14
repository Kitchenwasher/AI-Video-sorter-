import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


SCRIPT_PATH = Path(__file__).with_name("sort_videos_by_female_faces_gpu.py")


def _write_small_video(path: Path) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 6.0, (48, 36))
    if not writer.isOpened():
        raise RuntimeError(f"failed to create {path}")
    try:
        for idx in range(10):
            frame = np.zeros((36, 48, 3), dtype=np.uint8)
            frame[:] = (40 + idx, 90, 170)
            writer.write(frame)
    finally:
        writer.release()


class DuplicatesCliTests(unittest.TestCase):
    def _run_backend(self, args: list[str], cwd: Path) -> dict:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), *args],
            capture_output=True,
            text=True,
            cwd=str(cwd),
            check=False,
        )
        stdout = (completed.stdout or "").strip()
        payload = None
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except Exception:
                continue
            if isinstance(parsed, dict):
                payload = parsed
                break
        if payload is None:
            raise AssertionError(f"No JSON payload from backend. stdout={stdout}\nstderr={completed.stderr}")
        if completed.returncode != 0:
            raise AssertionError(f"Backend failed: {payload}")
        return payload

    def test_duplicates_scan_and_apply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "sorted"
            folder = output_dir / "Female_D"
            folder.mkdir(parents=True, exist_ok=True)
            a = folder / "a.avi"
            b = folder / "b.avi"
            _write_small_video(a)
            shutil.copyfile(a, b)

            scan_payload = self._run_backend(
                ["--output-dir", str(output_dir), "--duplicates-scan-json"],
                cwd=root,
            )
            self.assertTrue(scan_payload.get("ok", False))
            groups = scan_payload.get("groups", [])
            self.assertGreaterEqual(len(groups), 1)
            group = groups[0]
            duplicates = group.get("duplicate_paths", [])
            self.assertTrue(isinstance(duplicates, list) and len(duplicates) >= 1)

            apply_payload = self._run_backend(
                [
                    "--output-dir",
                    str(output_dir),
                    "--duplicates-apply-json",
                    json.dumps({"paths": duplicates}),
                ],
                cwd=root,
            )
            self.assertTrue(apply_payload.get("ok", False))
            details = apply_payload.get("details", {})
            self.assertGreaterEqual(int(details.get("moved_count", 0)), 1)
            self.assertFalse((output_dir / ".reports").exists())


if __name__ == "__main__":
    unittest.main()
