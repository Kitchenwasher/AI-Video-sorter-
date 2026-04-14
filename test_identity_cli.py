import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).with_name("sort_videos_by_female_faces_gpu.py")


class IdentityCliTests(unittest.TestCase):
    def _run_backend(self, args: list[str], cwd: Path) -> dict:
        command = [sys.executable, str(SCRIPT_PATH), *args]
        completed = subprocess.run(
            command,
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
            raise AssertionError(f"No JSON payload returned. stdout={stdout}\nstderr={completed.stderr}")
        if completed.returncode != 0:
            raise AssertionError(f"Backend failed: {payload}")
        return payload

    def test_identity_cli_list_and_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "sorted"
            output_dir.mkdir(parents=True, exist_ok=True)
            female_dir = output_dir / "Female_CLI"
            female_dir.mkdir(parents=True, exist_ok=True)
            (female_dir / "clip.mp4").write_bytes(b"clip")

            memory_file = root / "memory_cli.json"

            list_payload = self._run_backend(
                [
                    "--output-dir",
                    str(output_dir),
                    "--learning-memory-file",
                    str(memory_file),
                    "--identity-list-json",
                ],
                cwd=root,
            )
            self.assertTrue(list_payload.get("ok", False))
            self.assertEqual(int(list_payload.get("count", 0)), 1)

            lock_payload = self._run_backend(
                [
                    "--output-dir",
                    str(output_dir),
                    "--learning-memory-file",
                    str(memory_file),
                    "--identity-action",
                    "lock",
                    "--identity-folder",
                    "Female_CLI",
                ],
                cwd=root,
            )
            self.assertTrue(lock_payload.get("ok", False))
            self.assertEqual(lock_payload.get("action"), "lock")

            summary_payload = self._run_backend(
                [
                    "--output-dir",
                    str(output_dir),
                    "--learning-memory-file",
                    str(memory_file),
                    "--learning-summary-json",
                ],
                cwd=root,
            )
            self.assertTrue(summary_payload.get("ok", False))
            self.assertIn("items", summary_payload)
            self.assertFalse((output_dir / ".reports").exists())


if __name__ == "__main__":
    unittest.main()
