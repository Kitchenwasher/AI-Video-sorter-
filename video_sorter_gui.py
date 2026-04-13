#!/usr/bin/env python3
"""
Tkinter GUI for the ROCm female-face video sorter.

This GUI launches the backend sorter script, lets you choose source and
destination folders, and streams log output into the window.
"""

from __future__ import annotations

import queue
import json
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


APP_TITLE = "Female Face Video Sorter"
BACKEND_SCRIPT = Path(__file__).with_name("sort_videos_by_female_faces_gpu.py")
SETTINGS_FILE = Path(__file__).with_name("video_sorter_settings.json")


class VideoSorterGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("860x680")
        self.root.minsize(760, 560)

        self.process: subprocess.Popen[str] | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.reader_thread: threading.Thread | None = None

        self.source_var = tk.StringVar()
        self.destination_var = tk.StringVar()
        self.recursive_var = tk.BooleanVar(value=True)
        self.include_generated_var = tk.BooleanVar(value=False)
        self.include_generated_button_var = tk.StringVar()
        self.live_trace_var = tk.BooleanVar(value=False)
        self.live_trace_button_var = tk.StringVar()
        self.max_seconds_var = tk.StringVar(value="60")
        self.sample_every_var = tk.StringVar(value="2.0")
        self.stabilization_var = tk.StringVar(value="8.0")
        self.resize_width_var = tk.StringVar(value="960")
        self.max_workers_var = tk.StringVar(value="1")
        self.status_var = tk.StringVar(value="Ready")
        self.runtime_var = tk.StringVar(value="Runtime: checking...")
        self.live_status_var = tk.StringVar(value="Live: idle")
        self.progress_status_var = tk.StringVar(value="Progress: 0/0 videos")

        self._load_settings()
        self._update_include_generated_button_text()
        self._update_live_trace_button_text()
        self._build_ui()
        self._refresh_runtime_status()
        self.root.after(100, self._poll_log_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)

        top = ttk.Frame(self.root, padding=14)
        top.grid(row=0, column=0, sticky="nsew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Source Folder").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=(0, 8))
        ttk.Entry(top, textvariable=self.source_var).grid(row=0, column=1, sticky="ew", pady=(0, 8))
        ttk.Button(top, text="Browse", command=self._choose_source).grid(row=0, column=2, padx=(10, 0), pady=(0, 8))

        ttk.Label(top, text="Destination Folder").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(0, 8))
        ttk.Entry(top, textvariable=self.destination_var).grid(row=1, column=1, sticky="ew", pady=(0, 8))
        ttk.Button(top, text="Browse", command=self._choose_destination).grid(
            row=1, column=2, padx=(10, 0), pady=(0, 8)
        )

        runtime_bar = ttk.Frame(self.root, padding=(14, 0, 14, 10))
        runtime_bar.grid(row=1, column=0, sticky="ew")
        runtime_bar.columnconfigure(0, weight=1)

        ttk.Label(runtime_bar, textvariable=self.runtime_var).grid(row=0, column=0, sticky="w")
        ttk.Button(runtime_bar, text="Refresh Runtime", command=self._refresh_runtime_status).grid(
            row=0, column=1, padx=(10, 0)
        )

        options = ttk.LabelFrame(self.root, text="Options", padding=14)
        options.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 10))
        for idx in range(4):
            options.columnconfigure(idx, weight=1)

        ttk.Checkbutton(
            options,
            text="Scan child folders recursively",
            variable=self.recursive_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        ttk.Button(
            options,
            textvariable=self.include_generated_button_var,
            command=self._toggle_include_generated,
        ).grid(row=0, column=2, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Button(
            options,
            textvariable=self.live_trace_button_var,
            command=self._toggle_live_trace,
        ).grid(row=1, column=2, columnspan=2, sticky="ew", pady=(0, 10))

        ttk.Label(options, text="Max Seconds").grid(row=1, column=0, sticky="w")
        ttk.Entry(options, textvariable=self.max_seconds_var, width=12).grid(row=2, column=0, sticky="ew", padx=(0, 10))

        ttk.Label(options, text="Sample Every (sec)").grid(row=1, column=1, sticky="w")
        ttk.Entry(options, textvariable=self.sample_every_var, width=12).grid(row=2, column=1, sticky="ew", padx=(0, 10))

        ttk.Label(options, text="Stabilization (sec)").grid(row=3, column=2, sticky="w")
        ttk.Entry(options, textvariable=self.stabilization_var, width=12).grid(
            row=4, column=2, sticky="ew", padx=(0, 10)
        )

        ttk.Label(options, text="Resize Width").grid(row=3, column=3, sticky="w")
        ttk.Entry(options, textvariable=self.resize_width_var, width=12).grid(row=4, column=3, sticky="ew")

        ttk.Label(options, text="Max Workers").grid(row=5, column=0, sticky="w", pady=(10, 0))
        ttk.Entry(options, textvariable=self.max_workers_var, width=12).grid(row=6, column=0, sticky="ew", pady=(0, 2))

        ttk.Label(
            options,
            text="Source videos are discovered by extension and sorted from child folders too when recursive scan is enabled.",
            wraplength=760,
            foreground="#444444",
        ).grid(row=7, column=0, columnspan=4, sticky="w", pady=(12, 0))

        log_frame = ttk.LabelFrame(self.root, text="Logs", padding=10)
        log_frame.grid(row=3, column=0, sticky="nsew", padx=14, pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, wrap="word", height=20, state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)
        ttk.Label(log_frame, textvariable=self.live_status_var).grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Label(log_frame, textvariable=self.progress_status_var).grid(row=2, column=0, sticky="w", pady=(4, 0))

        footer = ttk.Frame(self.root, padding=(14, 0, 14, 14))
        footer.grid(row=4, column=0, sticky="ew")
        footer.columnconfigure(1, weight=1)
        footer.columnconfigure(2, weight=1)

        self.start_button = ttk.Button(footer, text="Start Sorting", command=self._start_sorting)
        self.start_button.grid(row=0, column=0, padx=(0, 8))

        self.stop_button = ttk.Button(footer, text="Stop", command=self._stop_sorting, state="disabled")
        self.stop_button.grid(row=0, column=1, sticky="w")

        self.progress = ttk.Progressbar(footer, mode="determinate", maximum=100, value=0)
        self.progress.grid(row=0, column=2, sticky="ew", padx=12)

        ttk.Label(footer, textvariable=self.status_var).grid(row=0, column=3, sticky="e")

    def _choose_source(self) -> None:
        selected = filedialog.askdirectory(title="Choose Source Folder")
        if selected:
            self.source_var.set(selected)
            self._save_settings()

    def _choose_destination(self) -> None:
        selected = filedialog.askdirectory(title="Choose Destination Folder")
        if selected:
            self.destination_var.set(selected)
            self._save_settings()

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _update_include_generated_button_text(self) -> None:
        if self.include_generated_var.get():
            self.include_generated_button_var.set("Generated Folders: Included")
        else:
            self.include_generated_button_var.set("Generated Folders: Skipped")

    def _update_live_trace_button_text(self) -> None:
        if self.live_trace_var.get():
            self.live_trace_button_var.set("Live Trace: On")
        else:
            self.live_trace_button_var.set("Live Trace: Off")

    def _toggle_include_generated(self) -> None:
        self.include_generated_var.set(not self.include_generated_var.get())
        self._update_include_generated_button_text()
        self._save_settings()

    def _toggle_live_trace(self) -> None:
        self.live_trace_var.set(not self.live_trace_var.get())
        self._update_live_trace_button_text()
        self._save_settings()

    def _load_settings(self) -> None:
        if not SETTINGS_FILE.exists():
            return
        try:
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return

        self.source_var.set(str(data.get("source_folder", "")))
        self.destination_var.set(str(data.get("destination_folder", "")))
        self.recursive_var.set(bool(data.get("recursive", True)))
        self.include_generated_var.set(bool(data.get("include_generated_folders", False)))
        self.live_trace_var.set(bool(data.get("live_trace", False)))
        self._update_include_generated_button_text()
        self._update_live_trace_button_text()
        self.max_seconds_var.set(str(data.get("max_seconds", "60")))
        self.sample_every_var.set(str(data.get("sample_every_sec", "2.0")))
        self.stabilization_var.set(str(data.get("stabilization_seconds", "8.0")))
        self.resize_width_var.set(str(data.get("resize_width", "960")))
        self.max_workers_var.set(str(data.get("max_workers", "1")))

    def _save_settings(self) -> None:
        data = {
            "source_folder": self.source_var.get().strip(),
            "destination_folder": self.destination_var.get().strip(),
            "recursive": self.recursive_var.get(),
            "include_generated_folders": self.include_generated_var.get(),
            "live_trace": self.live_trace_var.get(),
            "max_seconds": self.max_seconds_var.get().strip(),
            "sample_every_sec": self.sample_every_var.get().strip(),
            "stabilization_seconds": self.stabilization_var.get().strip(),
            "resize_width": self.resize_width_var.get().strip(),
            "max_workers": self.max_workers_var.get().strip(),
        }
        try:
            SETTINGS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as exc:
            self._append_log(f"[WARN] Failed to save settings: {exc}\n")

    def _refresh_runtime_status(self) -> None:
        if not BACKEND_SCRIPT.exists():
            self.runtime_var.set("Runtime: backend script not found")
            return

        try:
            command = [
                sys.executable,
                str(BACKEND_SCRIPT),
                "--input-dir",
                ".",
                "--output-dir",
                ".",
                "--print-runtime-json",
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=str(BACKEND_SCRIPT.parent),
                check=True,
            )
            info = json.loads(completed.stdout.strip())
            acceleration = info.get("acceleration", "Unknown")
            reason = info.get("reason", "")
            device_names = info.get("device_names") or []
            if device_names:
                device_text = ", ".join(device_names)
                self.runtime_var.set(f"Runtime: {acceleration} | {device_text}")
            else:
                self.runtime_var.set(f"Runtime: {acceleration} | {reason}")
        except Exception as exc:
            if isinstance(exc, subprocess.CalledProcessError):
                detail = (exc.stderr or exc.stdout or str(exc)).strip().splitlines()[-1]
                self.runtime_var.set(f"Runtime: check failed | {detail}")
            else:
                self.runtime_var.set(f"Runtime: check failed ({exc})")

    def _validate_inputs(self) -> list[str] | None:
        source_text = self.source_var.get().strip()
        destination_text = self.destination_var.get().strip()
        source = Path(source_text).expanduser()
        destination = Path(destination_text).expanduser()

        if not source.is_dir():
            messagebox.showerror(APP_TITLE, "Please choose a valid source folder.")
            return None
        if not destination_text:
            messagebox.showerror(APP_TITLE, "Please choose a destination folder.")
            return None
        if not BACKEND_SCRIPT.exists():
            messagebox.showerror(APP_TITLE, f"Backend script not found:\n{BACKEND_SCRIPT}")
            return None

        try:
            max_seconds = int(float(self.max_seconds_var.get().strip()))
            sample_every = float(self.sample_every_var.get().strip())
            stabilization = float(self.stabilization_var.get().strip())
            resize_width = int(float(self.resize_width_var.get().strip()))
            max_workers = int(float(self.max_workers_var.get().strip()))
        except ValueError:
            messagebox.showerror(APP_TITLE, "Numeric options must contain valid numbers.")
            return None

        if max_seconds <= 0 or sample_every <= 0 or stabilization <= 0 or resize_width <= 0 or max_workers <= 0:
            messagebox.showerror(APP_TITLE, "Numeric options must be greater than zero.")
            return None

        command = [
            sys.executable,
            str(BACKEND_SCRIPT),
            "--input-dir",
            str(source.resolve()),
            "--output-dir",
            str(destination.resolve()),
            "--max-seconds",
            str(max_seconds),
            "--sample-every-sec",
            str(sample_every),
            "--stabilization-seconds",
            str(stabilization),
            "--resize-width",
            str(resize_width),
            "--max-workers",
            str(max_workers),
        ]
        if not self.recursive_var.get():
            command.append("--no-recursive")
        if self.include_generated_var.get():
            command.append("--include-generated-folders")
        if self.live_trace_var.get():
            command.append("--live-trace")
        return command

    def _start_sorting(self) -> None:
        if self.process is not None:
            return

        command = self._validate_inputs()
        if command is None:
            return

        self._save_settings()
        self._append_log(f"$ {' '.join(command)}\n\n")
        self.status_var.set("Sorting in progress...")
        self.live_status_var.set("Live: starting...")
        self.progress_status_var.set("Progress: 0/0 videos")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.progress.configure(value=0, maximum=100)

        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(BACKEND_SCRIPT.parent),
        )
        self.reader_thread = threading.Thread(target=self._read_process_output, daemon=True)
        self.reader_thread.start()

    def _read_process_output(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.log_queue.put(line)
        return_code = self.process.wait()
        self.log_queue.put(f"\nProcess finished with exit code {return_code}\n")
        self.log_queue.put("__PROCESS_DONE__")

    def _poll_log_queue(self) -> None:
        while True:
            try:
                item = self.log_queue.get_nowait()
            except queue.Empty:
                break

            if item == "__PROCESS_DONE__":
                self.process = None
                self.reader_thread = None
                self.start_button.configure(state="normal")
                self.stop_button.configure(state="disabled")
                self.status_var.set("Finished")
                self.live_status_var.set("Live: idle")
            else:
                if item.startswith("[TRACE] "):
                    self.live_status_var.set(item.replace("[TRACE] ", "Live: ").strip())
                if item.startswith("[PROGRESS] "):
                    payload = item[len("[PROGRESS] ") :].strip().split()
                    values: dict[str, int] = {}
                    for chunk in payload:
                        if "=" not in chunk:
                            continue
                        key, raw = chunk.split("=", 1)
                        try:
                            values[key] = int(raw)
                        except ValueError:
                            continue
                    total = max(1, values.get("total", 0))
                    done = values.get("done", 0)
                    female = values.get("female", 0)
                    no_female = values.get("no_female", 0)
                    errors = values.get("errors", 0)
                    self.progress.configure(maximum=total, value=min(done, total))
                    self.progress_status_var.set(
                        f"Progress: {done}/{total} videos | female={female} | no_female={no_female} | errors={errors}"
                    )
                self._append_log(item)

        self.root.after(100, self._poll_log_queue)

    def _stop_sorting(self) -> None:
        if self.process is None:
            return
        self._append_log("\nStopping process...\n")
        self.process.terminate()
        self.status_var.set("Stopping...")

    def _on_close(self) -> None:
        if self.process is not None:
            if not messagebox.askyesno(APP_TITLE, "Sorting is still running. Close anyway and stop it?"):
                return
            self.process.terminate()
        self._save_settings()
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    ttk.Style(root).theme_use("clam")
    VideoSorterGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
